import torch
from gensim.models.keyedvectors import Word2VecKeyedVectors
from .cpg import Node, Method
from .cpg.types import TYPES
from dgl import DGLGraph
from typing import List
from .embeddings import node_embed


MAX_NODES = 300

edge_map_full = {
    'AST': 1,
    'CFG': 2,
    'CDG': 3,
    'DDG': 4
}


def _arrange_nodes_adj(method: Method):
    """
    extract nodes in function
    arrange adjacency matrix
    """
    node_set = set()
    edge_map = {"AST": [], "CFG": [], "CDG": [], "DDG": []}
    nodelist = sorted(list(method.node_id_set))
    hubs = {}
    for nid in nodelist:
        if method.nodes[nid].code == None or method.nodes[nid].code == '':
            continue
        if method.nodes[nid].node_type & TYPES.CDG == 0 and method.nodes[nid].node_type & TYPES.DDG == 0 and method.nodes[nid].node_type & TYPES.CFG == 0:
            pid = method.get_hub_node(nid)
            if pid not in hubs:
                hubs[nid] = [nid]
            else:
                hubs[pid].append(nid)
        else:
            hubs[nid] = [nid]
    node_list = sorted(list(hubs.keys()))
    for nid in node_list:
        # ast edges
        if nid in method.ast_edges:
            for ae in method.ast_edges[nid]:
                if ae in node_list:
                    edge_map["AST"].append((nid, ae))
        # refine ast edges
        if len(hubs[nid]) > 1:
            for pid in hubs[nid][1:]:
                if pid in method.ast_edges:
                    for ae in method.ast_edges[pid]:
                        if ae not in hubs[nid] and ae in node_list:
                            edge_map["AST"].append((nid, ae))
        # cfg edges
        if nid in method.cfg_edges:
            for fe in method.cfg_edges[nid]:
                if fe in node_list:
                    edge_map["CFG"].append((nid, fe))
        # cdg edges
        if nid in method.cdg_edges:
            for cde in method.cdg_edges[nid]:
                if cde in node_list:
                    edge_map["CDG"].append((nid, cde))
        # ddg edges
        if nid in method.ddg_edges:
            for dde in method.ddg_edges[nid]:
                if dde in node_list:
                    edge_map["DDG"].append((nid, dde))
    node_list = []
    for nid in sorted(list(hubs.keys())):
        nodes = [method.nodes[pid] for pid in hubs[nid]]
        node_list.append(nodes)
    return node_list, edge_map


class FPG:
    """
    A class for function program graph structure
    """

    def __init__(self, name, method: Method, label, code=None):
        self.name = name
        self.code = code
        self.node_list, self.edge_map = _arrange_nodes_adj(method)

        # assert len(self.node_list) > 0

        if len(self.node_list) > MAX_NODES:
            self.node_list = self.node_list[:MAX_NODES]
        self.label = label

    def embed(self, node_dim, wv):
        node_map = {}
        gInput = dict()
        gInput["targets"] = list()
        gInput["graph"] = list()
        gInput["node_features"] = list()
        gInput["targets"].append([self.label])
        gInput["code"] = self.code
        node_idx = 0
        for node in self.node_list:
            nid = node[0].id
            node_feature = node_embed(node, wv, node_dim)
            gInput['node_features'].append(node_feature)
            node_map[nid] = node_idx
            node_idx += 1
        for eType in self.edge_map:
            e_list = self.edge_map[eType]
            for e_out, e_in in e_list:
                if not e_out in node_map.keys() or not e_in in node_map.keys():
                    continue
                edge = [node_map[e_out], edge_map_full[eType], node_map[e_in]]
                gInput['graph'].append(edge)
        if len(gInput['node_features']) == 0 or len(gInput['graph']) == 0:
            return None
        gInput['file_name'] = self.name
        return gInput

    def __str__(self):
        r"""
        Print necessary information of this slice program graph
        """
        node_info = ""
        edge_info = ""
        for node in self.node_list:
            node_info += f'{node.id} = ({node.node_attr}, "{node.code}", {node.line_number})\n'
        for et in self.edge_map:
            for e in self.edge_map[et]:
                edge_info += f"{e[0]} -> {e[1]}, type = {et}\n"
        return f"Filename: {self.name}\nLabel: {self.label}\n" + node_info + edge_info
