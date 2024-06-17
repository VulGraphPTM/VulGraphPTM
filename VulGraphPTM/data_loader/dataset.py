import copy
import json
import sys
import re

import torch
import dgl
from dgl import DGLGraph, batch
from tqdm import tqdm

from data_loader.batch_graph import GGNNBatchGraph
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from utils import load_default_identifiers, initialize_batch, debug


keywords = ["alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
            "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
            "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
            "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
            "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
            "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
            "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
            "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "reflexpr",
            "register", "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
            "static_assert", "static_cast", "struct", "switch", "synchronized", "template", "this",
            "thread_local", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned",
            "using", "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq", "NULL"]
brackets = [',', ':', ';', '{', '}', '[', ']', '(', ')', '::', '/', '//', '/*', '*/']
ops = ['=', '+', '-', '*', '/', '+=', '-=', '*=', '/=', '>', '<', '>=', '<=', '==', '!=', '<<', '>>', '<<=', '>>=', '!', '&&', '||', '->', '.',
       '?', '++', '--', '&', '|', '^', '&=', '|=', '^=']


def convert_examples_to_features(func, tokenizer, block_size=512):
    if func == None or tokenizer == None:
        return None, None
    func_lines = str(func).split('\n')
    code_tokens = []
    line_masks = {}
    for ids, line in enumerate(func_lines):
        line = line + '\n'
        line_tokens = tokenizer.tokenize(line)
        line_map = [(tk, idx+1+len(code_tokens)) for idx, tk in enumerate(line_tokens)]
        code_tokens += line_tokens
        line_masks[ids + 1] = line_map
        code_tokens = code_tokens[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    source_tokens += [tokenizer.pad_token] * padding_length
    assert len(source_tokens) == len(source_ids)
    return source_tokens, source_ids, line_masks


def is_valid_variable_name(variable_name):
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    match = re.match(pattern, variable_name)
    return match is not None


def remove_special_characters(string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9_]', '', string)
    return cleaned_string


def check_is_valid(token: str):
    var = token.strip().replace('\u0120', '').replace('\u010a', '').replace('\u0109', '')
    if is_valid_variable_name(var) and var not in keywords:
        return True
    elif remove_special_characters(var) in keywords:
        return True
    elif var in ops:
        return True
    return False


def node_token_map(node_line, node_code, src_tokens, line_masks, tokenizer):
    node_ids_map = []
    for nln, lcd in zip(node_line, node_code):
        line = lcd + '\n'
        line_tokens = tokenizer.tokenize(line)
        cur_ids = []
        if nln in line_masks:
            for ltk in line_masks[nln]:
                tk, lid = ltk
                if check_is_valid(tk) and tk in line_tokens and lid < 512:
                    cur_ids.append(lid)
        node_ids_map.append(cur_ids)
    assert len(node_ids_map) == len(node_line)
    return node_ids_map


class DataEntry:
    def __init__(self, datset, num_nodes, features, edges, target, code=None, node_line=None, node_code=None, tokenizer=None):
        self.dataset = datset
        self.num_nodes = num_nodes
        self.target = target
        self.code = code
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.source_tokens, self.source_ids, self.line_masks = convert_examples_to_features(code, tokenizer)
        node_ids_map = node_token_map(node_line, node_code, self.source_tokens, self.line_masks, tokenizer)
        if node_line:
            assert len(node_ids_map) == self.num_nodes == len(features)
        self.node_ids_map = node_ids_map
        # node_ids_map = torch.IntTensor(node_ids_map)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])})


class DataSet:
    def __init__(self, train_src, valid_src=None, test_src=None, batch_size=32, n_ident=None, g_ident=None, l_ident=None, add_self_loop=False, batch_type='lib', tokenizer=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.test_names = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.add_self_loop = add_self_loop
        self.batch_type = batch_type
        self.tokenizer = tokenizer
        self.n_ident, self.g_ident, self.l_ident= load_default_identifiers(n_ident, g_ident, l_ident)
        self.read_dataset(test_src, train_src, valid_src)
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, test_src, train_src, valid_src):
        debug('Reading Train File!')
        if train_src is not None:
            with open(train_src) as fp:
                train_data = json.load(fp)
                for entry in tqdm(train_data):
                    if "code" not in entry:
                        entry["code"] = None
                    if "node_line" not in entry:
                        entry["node_line"] = None
                        entry["node_code"] = None
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0], code=entry["code"], node_line=entry["node_line"], node_code=entry["node_code"], tokenizer=self.tokenizer)
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    self.train_examples.append(example)
        if valid_src is not None:
            debug('Reading Validation File!')
            with open(valid_src) as fp:
                valid_data = json.load(fp)
                for entry in tqdm(valid_data):
                    if "code" not in entry:
                        entry["code"] = None
                    if "node_line" not in entry:
                        entry["node_line"] = None
                        entry["node_code"] = None
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0], code=entry["code"], node_line=entry["node_line"], node_code=entry["node_code"], tokenizer=self.tokenizer)
                    self.valid_examples.append(example)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = json.load(fp)
                for entry in tqdm(test_data):
                    if "code" not in entry:
                        entry["code"] = None
                    if "node_line" not in entry:
                        entry["node_line"] = None
                        entry["node_code"] = None
                    example = DataEntry(datset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0], code=entry["code"], node_line=entry["node_line"], node_code=entry["node_code"], tokenizer=self.tokenizer)
                    if self.feature_size == 0:
                        self.feature_size = example.features.size(1)
                        debug('Feature Size %d' % self.feature_size)
                    self.test_examples.append(example)
                    self.test_names.append(entry['file_name'])

    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids, ret_name=False):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        if ret_name:
            names = [self.test_names[i] for i in ids]
            return batch_graph, torch.FloatTensor(labels), names
        return batch_graph, torch.FloatTensor(labels)
    
    def get_dataset_by_ids_to_batch(self, entries, ids, ret_name=False, add_self_loop=False):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        graphs = [copy.deepcopy(entry.graph) for entry in taken_entries]
        if add_self_loop:
            graphs = [dgl.add_self_loop(g) for g in graphs]
        if ret_name:
            names = [self.test_names[i] for i in ids]
            return batch(graphs), torch.FloatTensor(labels), names
        return batch(graphs), torch.FloatTensor(labels)
    
    def get_dataset_by_ids_to_batch_with_tokens(self, entries, ids, ret_name=False, add_self_loop=False):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        graphs = [copy.deepcopy(entry.graph) for entry in taken_entries]
        source_ids = [torch.tensor(e.source_ids) for e in taken_entries]
        node_ids_map = [e.node_ids_map for e in taken_entries]
        if add_self_loop:
            graphs = [dgl.add_self_loop(g) for g in graphs]
        if ret_name:
            names = [self.test_names[i] for i in ids]
            return batch(graphs), torch.FloatTensor(labels), torch.stack(source_ids), names, node_ids_map
        return batch(graphs), torch.FloatTensor(labels), torch.stack(source_ids), node_ids_map
    
    def train_to_dataloader(self):
        graph_list = [copy.deepcopy(entry.graph) for entry in self.train_examples]
        label_list = [entry.target for entry in self.train_examples]
        batch_dataset = BatchDataset("BatchDataset", graph_list, label_list, self.batch_size, self.train_batches)
        return batch_dataset_to_dataloader(batch_dataset)
    
    def valid_to_dataloader(self):
        graph_list = [copy.deepcopy(entry.graph) for entry in self.valid_examples]
        label_list = [entry.target for entry in self.valid_examples]
        batch_dataset = BatchDataset("BatchDataset", graph_list, label_list, self.batch_size, self.valid_batches)
        return batch_dataset_to_dataloader(batch_dataset)
    
    def test_to_dataloader(self):
        graph_list = [copy.deepcopy(entry.graph) for entry in self.test_examples]
        label_list = [entry.target for entry in self.test_examples]
        batch_dataset = BatchDataset("BatchDataset", graph_list, label_list, self.batch_size, self.test_batches)
        return batch_dataset_to_dataloader(batch_dataset)

    def get_next_train_batch(self, with_tokens=False):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        if self.batch_type == 'origin':
            return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)
        if with_tokens:
            return self.get_dataset_by_ids_to_batch_with_tokens(self.train_examples, ids, add_self_loop=self.add_self_loop)
        return self.get_dataset_by_ids_to_batch(self.train_examples, ids, add_self_loop=self.add_self_loop)

    def get_next_valid_batch(self, with_tokens=False):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        if self.batch_type == 'origin':
            return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)
        if with_tokens:
            return self.get_dataset_by_ids_to_batch_with_tokens(self.valid_examples, ids, add_self_loop=self.add_self_loop)
        return self.get_dataset_by_ids_to_batch(self.valid_examples, ids, add_self_loop=self.add_self_loop)

    def get_next_test_batch(self, ret_name=False, with_tokens=False):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        if self.batch_type == 'origin':
            return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids, ret_name)
        if with_tokens:
            return self.get_dataset_by_ids_to_batch_with_tokens(self.test_examples, ids, ret_name, add_self_loop=self.add_self_loop)
        return self.get_dataset_by_ids_to_batch(self.test_examples, ids, ret_name, add_self_loop=self.add_self_loop)


class BatchDataset(DGLDataset):
    def __init__(self, dataset:str, graph_list, label_list, batch_size, batch_ids_list=None):
        self.graph_list = graph_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.batch_ids_list = batch_ids_list

        if batch_ids_list:
            self._re_sample_by_ids()

        super(BatchDataset, self).__init__(dataset)
    
    def _re_sample_by_ids(self):
        new_graphs = []
        new_labels = []
        while len(self.batch_ids_list) > 0:
            batch_ids = self.batch_ids_list.pop()
            taken_entries = [self.graph_list[i] for i in batch_ids]
            taken_labels = [self.label_list[i] for i in batch_ids]
            new_graphs += taken_entries
            new_labels += taken_labels
        assert len(new_graphs) == len(self.graph_list)
        self.graph_list = new_graphs
        self.label_list = new_labels
    
    def __getitem__(self, idx):
        return self.graph_list[idx], self.label_list[idx]
    
    def __len__(self):
        assert len(self.graph_list) == len(self.label_list)
        return len(self.graph_list)


def batch_dataset_to_dataloader(dataset: BatchDataset):
    return GraphDataLoader(
        dataset, batch_size=dataset.batch_size, shuffle=False, num_workers=4
    )
