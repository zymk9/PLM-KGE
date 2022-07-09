import os
import json
import torch
import torch.utils.data.dataset
import warnings
import numpy as np

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer, get_concept_dict
from logger_config import logger

entity_dict = get_entity_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc) if not args.no_desc else head_word
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc) if not args.no_desc else tail_word
        tail_encoded_inputs = _custom_tokenize(text=tail_text)

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


class TripletDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, mode, ns_size):
        self.path_list = path.split(',')
        self.mode = mode
        self.negative_sample_size = ns_size
        self.nentity = len(get_concept_dict().ent2idx)

        assert all(os.path.exists(path) for path in self.path_list)
        self.examples = []
        for path in self.path_list:
            if not self.examples:
                self.examples = load_data(path)
            else:
                self.examples.extend(load_data(path))

        self.count_frequency()
        self.get_true_head_and_tail()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        positive_sample = self.examples[index]
        head = positive_sample.head_id
        tail = positive_sample.tail_id
        relation = positive_sample.relation

        subsampling_weight = self.count[(head, relation)] + self.count[(relation, tail)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0
        if self.mode == "head-batch":
            e_filter = self.concept_filter_h(head, relation)
        if self.mode == "tail-batch":
            e_filter = self.concept_filter_t(tail, relation)
        if len(e_filter) > 0:
            ns_size = min(self.negative_sample_size, len(e_filter))
            negative_sample = np.random.choice(e_filter, ns_size)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size = negative_sample.size

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size)

            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        if self.mode == 'head-batch':
            neg_ex = [Example(head, relation, tail) for head in np.nditer(negative_sample)]
        else:
            neg_ex = [Example(head, relation, tail) for tail in np.nditer(negative_sample)]
        
        neg_vec = [ex.vectorize() for ex in neg_ex]
            
        return positive_sample.vectorize(), neg_vec, subsampling_weight, self.negative_sample_size

    def count_frequency(self, start=4):
        self.count = {}
        for ex in self.examples:
            head = ex.head_id
            tail = ex.tail_id
            relation = ex.relation
            if (head, relation) not in self.count:
                self.count[(head, relation)] = start
            else:
                self.count[(head, relation)] += 1

            if (relation, tail) not in self.count:
                self.count[(relation, tail)] = start
            else:
                self.count[(relation, tail)] += 1

    def get_true_head_and_tail(self):
        self.true_head = {}
        self.true_tail = {}

        for ex in self.examples:
            head = ex.head_id
            tail = ex.tail_id
            relation = ex.relation
            if (head, relation) not in self.true_tail:
                self.true_tail[(head, relation)] = []
            self.true_tail[(head, relation)].append(tail)
            if (relation, tail) not in self.true_head:
                self.true_head[(relation, tail)] = []
            self.true_head[(relation, tail)].append(head)

        for relation, tail in self.true_head:
            self.true_head[(relation, tail)] = np.array(list(set(self.true_head[(relation, tail)])))
        for head, relation in self.true_tail:
            self.true_tail[(head, relation)] = np.array(list(set(self.true_tail[(head, relation)])))

    @staticmethod
    def collate_fn(batch_data):
        pos_ex = [x[0] for x in batch_data]
        neg_ex = []
        for x in batch_data:
            neg_ex += x[1]

        subsampling_weight = torch.tensor([x[2] for x in batch_data])

        pos_hr_token_ids, pos_hr_mask = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_ids']) for ex in pos_ex],
            pad_token_id=get_tokenizer().pad_token_id)
        pos_hr_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_type_ids']) for ex in pos_ex],
            need_mask=False)

        pos_tail_token_ids, pos_tail_mask = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_ids']) for ex in pos_ex],
            pad_token_id=get_tokenizer().pad_token_id)
        pos_tail_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_type_ids']) for ex in pos_ex],
            need_mask=False)

        pos_head_token_ids, pos_head_mask = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_ids']) for ex in pos_ex],
            pad_token_id=get_tokenizer().pad_token_id)
        pos_head_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_type_ids']) for ex in pos_ex],
            need_mask=False)

        neg_hr_token_ids, neg_hr_mask = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_ids']) for ex in neg_ex],
            pad_token_id=get_tokenizer().pad_token_id)
        neg_hr_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['hr_token_type_ids']) for ex in neg_ex],
            need_mask=False)

        neg_tail_token_ids, neg_tail_mask = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_ids']) for ex in neg_ex],
            pad_token_id=get_tokenizer().pad_token_id)
        neg_tail_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['tail_token_type_ids']) for ex in neg_ex],
            need_mask=False)

        neg_head_token_ids, neg_head_mask = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_ids']) for ex in neg_ex],
            pad_token_id=get_tokenizer().pad_token_id)
        neg_head_token_type_ids = to_indices_and_mask(
            [torch.LongTensor(ex['head_token_type_ids']) for ex in neg_ex],
            need_mask=False)

        batch_exs = [ex['obj'] for ex in batch_data]

        batch_dict = {
            'pos_hr_token_ids': pos_hr_token_ids,
            'pos_hr_mask': pos_hr_mask,
            'pos_hr_token_type_ids': pos_hr_token_type_ids,
            'pos_tail_token_ids': pos_tail_token_ids,
            'pos_tail_mask': pos_tail_mask,
            'pos_tail_token_type_ids': pos_tail_token_type_ids,
            'pos_head_token_ids': pos_head_token_ids,
            'pos_head_mask': pos_head_mask,
            'pos_head_token_type_ids': pos_head_token_type_ids,
            'neg_hr_token_ids': neg_hr_token_ids,
            'neg_hr_mask': neg_hr_mask,
            'neg_hr_token_type_ids': neg_hr_token_type_ids,
            'neg_tail_token_ids': neg_tail_token_ids,
            'neg_tail_mask': neg_tail_mask,
            'neg_tail_token_type_ids': neg_tail_token_type_ids,
            'neg_head_token_ids': neg_head_token_ids,
            'neg_head_mask': neg_head_mask,
            'neg_head_token_type_ids': neg_head_token_type_ids,
            'batch_data': batch_exs,
            'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
            'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        }

        return batch_dict


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]

    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
