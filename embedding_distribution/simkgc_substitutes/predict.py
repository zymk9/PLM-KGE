import os
import json
import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from doc import collate, Example, Dataset
from config import args
from models import build_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer
from logger_config import logger
from transformers import AutoModel

class BertPredictor:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False

    def load(self, ckt_path, use_data_parallel=False):
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_model(self.train_args)
        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
    
        self.model.eval()

        if use_data_parallel and torch.cuda.device_count() > 1:
            logger.info('Use data parallel predictor')
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.use_cuda = True
        elif torch.cuda.is_available():
            self.model.cuda()
            self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info('Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.use_link_graph = self.train_args.use_link_graph
        args.is_test = True

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=1,
            batch_size=max(args.batch_size, 512),
            collate_fn=collate,
            shuffle=False)

        hr_tensor_list, tail_tensor_list = [], []
        for idx, batch_dict in enumerate(data_loader):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            tail_tensor_list.append(outputs['tail_vector'])

        return torch.cat(hr_tensor_list, dim=0), torch.cat(tail_tensor_list, dim=0)

    @torch.no_grad()
    def predict_tensor(self, list_of_example, file_name):
        examples = []
        for ex in list_of_example:
            examples.append(Example(head_id=ex["head_id"], relation=ex["relation"], tail_id=ex["tail_id"]))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)
        hr_tensor_list = []
        t_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            hr_tensor_list.append(outputs['hr_vector'])
            t_tensor_list.append(outputs['tail_vector'])
        torch.save(torch.cat(hr_tensor_list, dim=0), file_name + '_hr.pt')
        torch.save(torch.cat(t_tensor_list, dim=0), file_name + '_t.pt')





    @torch.no_grad()
    def predict_hr(self) -> torch.tensor:
        ############ indataset hr ###########
        indataset_entities_file = open('/content/drive/MyDrive/SimKGC/data/FB15k237/indataset_ents.json', encoding="utf-8")
        indataset_entity_exs_json = json.load(indataset_entities_file)
        indataset_ents = [ent["entity_id"] for ent in indataset_entity_exs_json]
        examples = []

        for ent_id in indataset_ents:
            examples.append(Example(head_id=ent_id, relation='mode of transportation transportation how to get here travel destination travel ',
                                    tail_id=ent_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)
        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = False
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['hr_vector'])
        
        torch.save(torch.cat(ent_tensor_list, dim=0), 'indataset_hr.pt')
        ############ outdataset hr ############
        outdataset_entities_file = open('/content/drive/MyDrive/SimKGC/data/FB15k237/outdataset_ents.json', encoding="utf-8")
        outdataset_entity_exs_json = json.load(outdataset_entities_file)
        outdataset_ents = [ent["entity_id"] for ent in outdataset_entity_exs_json]
        examples = []
        print(outdataset_ents)

        for ent_id in outdataset_ents:
            examples.append(Example(head_id=ent_id, relation='mode of transportation transportation how to get here travel destination travel ',
                                    tail_id=ent_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)
        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = False
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['hr_vector'])
        
        torch.save(torch.cat(ent_tensor_list, dim=0), 'outdataset_hr.pt')
        return torch.cat(ent_tensor_list, dim=0)

    @torch.no_grad()
    def predict_tail_emb(self) -> torch.tensor:
        tail_entities_file = open('/content/drive/MyDrive/SimKGC/data/FB15k237/tail_ents.json', encoding="utf-8")
        tail_entity_exs_json = json.load(tail_entities_file)
        examples = []
        for entity_id in tail_entity_exs_json:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])
        torch.save(torch.cat(ent_tensor_list, dim=0), "tail.pt")
        return torch.cat(ent_tensor_list, dim=0)

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))
        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task),
            num_workers=2,
            batch_size=max(args.batch_size, 1024),
            collate_fn=collate,
            shuffle=False)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader)):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)
            outputs = self.model(**batch_dict)
            ent_tensor_list.append(outputs['ent_vectors'])

        return torch.cat(ent_tensor_list, dim=0)
