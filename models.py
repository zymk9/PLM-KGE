from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoAdapterModel, AutoConfig, AdapterConfig
from transformers.adapters.heads import ClassificationHead, PredictionHead

from triplet_mask import construct_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomHead(PredictionHead):
    def __init__(self, model, head_name, **config):
        super().__init__(head_name)
        self.config = config
        self.build(model=model)

    def forward(self, outputs, cls_output=None, attention_mask=None, return_dict=False, **kwargs):
        return _pool_output('mean', cls_output, attention_mask, outputs.last_hidden_state)


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.training_mode = args.training_mode

        self.hr_bert = AutoAdapterModel.from_pretrained(args.pretrained_model, config=self.config)
        self.tail_bert = deepcopy(self.hr_bert)

        if self.training_mode != 'adapterfusion':
            if 'concept_pred_hr' not in self.hr_bert.config.adapters:
                adapter_config = AdapterConfig.load('houlsby')
                self.hr_bert.add_adapter('concept_pred_hr', config=adapter_config)

            if 'link_pred_hr' not in self.hr_bert.config.adapters:
                adapter_config = AdapterConfig.load('houlsby')
                self.hr_bert.add_adapter('link_pred_hr', config=adapter_config)

            if 'concept_pred_t' not in self.tail_bert.config.adapters:
                adapter_config = AdapterConfig.load('houlsby')
                self.tail_bert.add_adapter('concept_pred_t', config=adapter_config)

            if 'link_pred_t' not in self.tail_bert.config.adapters:
                adapter_config = AdapterConfig.load('houlsby')
                self.tail_bert.add_adapter('link_pred_t', config=adapter_config)

        if self.training_mode == 'concept_pred':
            self.hr_bert.train_adapter(['concept_pred_hr'])
            self.tail_bert.train_adapter(['concept_pred_t'])

        elif self.training_mode == 'link_pred':
            self.hr_bert.train_adapter(['link_pred_hr'])
            self.tail_bert.train_adapter(['link_pred_t'])
        
        else:
            self.hr_bert.load_adapter(args.concept_pred_hr, with_head=False)
            self.tail_bert.load_adapter(args.concept_pred_t, with_head=False)
            self.hr_bert.load_adapter(args.link_pred_hr, with_head=False)
            self.tail_bert.load_adapter(args.link_pred_t, with_head=False)

            self.hr_bert.add_adapter_fusion(['concept_pred_hr', 'link_pred_hr'], 'dynamic')
            self.tail_bert.add_adapter_fusion(['concept_pred_t', 'link_pred_t'], 'dynamic')
            self.hr_bert.train_adapter_fusion([['concept_pred_hr', 'link_pred_hr']])
            self.tail_bert.train_adapter_fusion([['concept_pred_t', 'link_pred_t']])

        # For the concept prediction head
        if args.training_mode == 'concept_pred':
            self.hr_pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.hr_classifier = nn.Linear(self.config.hidden_size, args.num_concepts)
            self.hr_dropout = nn.Dropout(args.dropout)

            self.tail_pre_classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.tail_classifier = nn.Linear(self.config.hidden_size, args.num_concepts)
            self.tail_dropout = nn.Dropout(args.dropout)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def _pred_concept(self, encoder, pre_classifier, dropout, classifier, token_ids, mask, token_type_ids):
        cls_output = self._encode(encoder, token_ids, mask, token_type_ids)
        cls_output = pre_classifier(cls_output)
        cls_output = nn.ReLU()(cls_output)
        cls_output = dropout(cls_output)
        logits = classifier(cls_output)
        return logits

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        if self.training_mode == 'concept_pred':
            hr_logits = self._pred_concept(self.hr_bert, self.hr_pre_classifier, self.hr_dropout, self.hr_classifier,
                                           hr_token_ids, hr_mask, hr_token_type_ids)
            tail_logits = self._pred_concept(self.tail_bert, self.tail_pre_classifier, self.tail_dropout,
                                             self.tail_classifier, tail_token_ids, tail_mask, tail_token_type_ids)
            return {'hr_logits': hr_logits, 'tail_logits': tail_logits}

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
