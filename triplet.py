import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from logger_config import logger


@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])
        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str) -> set:
        return self.hr2tails.get((h, r), set())


class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class LinkGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


class ConceptDict:

    def __init__(self, concept_dict_dir: str):
        logger.info('Start to build concept dictionary from {}'.format(concept_dict_dir))
        self.rel2dom_h = {} # relation idx -> set(domain_head idx)
        self.rel2dom_t = {} # relation idx -> set(domain_tail idx)
        self.rel2nn = {}    # relation idx -> 0, 1, 2, 3 (1-1, 1-N, N-1, N-N)
        self.dom2ent = {}   # domain idx -> set(entity idx)
        self.ent2dom = {}   # entity idx -> set(domain idx)
        self.rel2idx = {}   # relation -> idx
        self.ent2idx = {}   # entity id -> idx
        self.idx2ent = {}   # idx -> entity id
        self.idx2rel = {}   # idx -> relation

        self._load(concept_dict_dir)
        self._add_inversed_relations()

    def _load(self, dir: str):
        with open(os.path.join(dir, 'rel2dom_h.json')) as f:
            self.rel2dom_h = json.load(f)
            self.rel2dom_h = {int(k): set(vals) for k, vals in self.rel2dom_h.items()}

        with open(os.path.join(dir, 'rel2dom_t.json')) as f:
            self.rel2dom_t = json.load(f)
            self.rel2dom_t = {int(k): set(vals) for k, vals in self.rel2dom_t.items()}

        with open(os.path.join(dir, 'rel2nn.json')) as f:
            self.rel2nn = json.load(f)
            self.rel2nn = {int(k): int(v) for k, v in self.rel2nn.items()}

        with open(os.path.join(dir, 'dom_ent.json')) as f:
            self.dom2ent = json.load(f)
            self.dom2ent = {int(k): set(vals) for k, vals in self.dom2ent.items()}

        with open(os.path.join(dir, 'ent_dom.json')) as f:
            self.ent2dom = json.load(f)
            self.ent2dom = {int(k): set(vals) for k, vals in self.ent2dom.items()}

        with open(os.path.join(dir, 'relations.dict')) as f:
            for line in f:
                rel_idx, rel = line.strip().split('\t')
                self.rel2idx[_normalize_fb15k237_relation(rel)] = int(rel_idx)  # consistent with SimKGC
                self.idx2rel[int(rel_idx)] = rel

        with open(os.path.join(dir, 'entities.dict')) as f:
            for line in f:
                ent_idx, ent = line.strip().split('\t')
                self.ent2idx[ent] = int(ent_idx)
                self.idx2ent[int(ent_idx)] = ent

    # Add concept data for inversed relations
    def _add_inversed_relations(self):
        cur_idx = max(self.rel2idx.values()) + 1
        inv_rel_dict = {}
        inv_idx_dict = {}
        for rel, idx in self.rel2idx.items():
            inv_rel = "inverse {}".format(rel)
            inv_rel_dict[inv_rel] = cur_idx
            inv_idx_dict[cur_idx] = inv_rel
            self.rel2dom_h[cur_idx] = self.rel2dom_t[idx]
            self.rel2dom_t[cur_idx] = self.rel2dom_h[idx]
            
            rel_type = self.rel2nn[idx]
            if rel_type == 1 or rel_type == 2:
                rel_type = 3 - rel_type
            self.rel2nn[cur_idx] = rel_type
            cur_idx += 1
        
        self.rel2idx.update(inv_rel_dict)
        self.idx2rel.update(inv_idx_dict)

    def get_rel_type(self, rel: str) -> int:
        return self.rel2nn[self.rel2idx[rel]]

    # Return duduced domain index for head entity
    def deduce_head_dom(self, head_id: str, rel: str) -> int:
        head_idx = self.ent2idx[head_id]
        rel_idx = self.rel2idx[rel]
        if head_idx not in self.ent2dom:
            return -1
        head_dom = self.ent2dom[head_idx]
        rel_dom = self.rel2dom_h[rel_idx]
        return next(iter(head_dom.intersection(rel_dom)))

    # Return duduced domain index for tail entity
    def duduce_tail_dom(self, tail_id: str, rel: str) -> int:
        tail_idx = self.ent2idx[tail_id]
        rel_idx = self.rel2idx[rel]
        if tail_idx not in self.ent2dom:
            return -1
        tail_dom = self.ent2dom[tail_idx]
        rel_dom = self.rel2dom_t[rel_idx]
        return next(iter(tail_dom.intersection(rel_dom)))

    def get_ent_idx(self, ent_id: str) -> int:
        return self.ent2idx[ent_id]

    def get_class_weights(self):
        return compute_class_weight('balanced', classes=np.array([0, 1, 2, 3]), 
            y=np.array([v for k, v in self.rel2nn.items()]))

    def concept_filter_h(self, head_id: str, relation: str):
        if str(relation) not in self.rel2idx:
            return []

        rel_idx = self.rel2idx[relation]
        rel_hc = self.rel2dom_h[rel_idx]
        set_hc = rel_hc
        h = []

        if self.rel2nn[rel_idx] == 0 or self.rel2nn[rel_idx] == 1:
            if head_id not in self.ent2idx:
                for hc in rel_hc:
                    for ent in self.conc_ents[hc]:
                        h.append(ent)
            else:
                for dom in self.ent2dom[self.ent2idx[head_id]]:
                    for ent in self.dom2ent[dom]:
                        h.append(ent)
        else:
            if head_id in self.ent2idx:
                set_ent_dom = self.ent2dom[self.ent2idx[head_id]]
            else:
                set_ent_dom = set([])
            set_diff = set_hc - set_ent_dom
            set_diff = list(set_diff)
            for dom in set_diff:
                for ent in self.dom2ent[dom]:
                    h.append(ent)

        h = set(h)
        return list(h)

    def concept_filter_t(self, tail_id: str, relation: str):
        if relation not in self.rel2idx:
            return []

        rel_idx = self.rel2idx[relation]
        rel_tc = self.rel2dom_t[rel_idx]
        set_tc = rel_tc
        t = []

        if self.rel2nn[rel_idx] == 0 or self.rel2nn[rel_idx] == 2:
            if tail_id in self.ent2idx:
                for dom in self.ent2dom[self.ent2idx[tail_id]]:
                    for ent in self.dom2ent[dom]:
                        t.append(ent)
            else:
                for tc in rel_tc:
                    for ent in self.dom2ent[tc]:
                        t.append(ent)
        else:
            if tail_id in self.ent2idx:
                set_ent_dom = self.ent2dom[self.ent2idx[tail_id]]
            else:
                set_ent_dom = set([])
            set_diff = set_tc - set_ent_dom
            set_diff = list(set_diff)
            for dom in set_diff:
                for ent in self.dom2ent[dom]:
                    t.append(ent)

        t = set(t)
        return list(t)


def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation
