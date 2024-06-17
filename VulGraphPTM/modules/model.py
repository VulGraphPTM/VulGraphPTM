import torch
import dgl
from dgl.nn import GatedGraphConv
from dgl.nn import RelGraphConv
from dgl.nn import GATConv
from torch import nn
import torch.nn.functional as f
from dgl.readout import sum_nodes, softmax_nodes


class GlobalAttentionPooling(nn.Module):
    """
    Global Attention Pooling from `Gated Graph Sequence Neural Networks`
    """
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
    
    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            if get_attention:
                return readout, gate
            else:
                return readout


class DevignModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(DevignModel, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(output_dim, output_dim, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(output_dim, output_dim, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_dim + output_dim
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False, device=None, output_ggnn=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda, device=device)
        outputs = self.ggnn(graph, features, edge_types)
        x_i, _ = batch.de_batchify_graphs(features)
        h_i, _ = batch.de_batchify_graphs(outputs)
        c_i = torch.cat((h_i, x_i), dim=-1)
        if output_ggnn:
            ggnn_sum = c_i.sum(dim=1)
            # print(ggnn_sum.size())
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        if Y_1.size()[2] < 2:
            raise ValueError(f'The dimension of {Y_1.size()} is too small.')
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result = self.sigmoid(avg).squeeze(dim=-1)
        if output_ggnn:
            return result, ggnn_sum
        else:
            return result


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False, device=None, output_ggnn=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda, device=device)
        outputs = self.ggnn(graph, features, edge_types)
        print(outputs.size())
        h_i, _ = batch.de_batchify_graphs(outputs)
        print(h_i.size())
        if output_ggnn:
            ggnn_sum = h_i.sum(dim=1)
            cl_sum = self.classifier(ggnn_sum)
            result = self.sigmoid(cl_sum).squeeze(dim=-1)
            return result, ggnn_sum
        else:
            ggnn_sum = h_i.sum(dim=1)
            print(ggnn_sum.size())
            ggnn_sum = self.classifier(ggnn_sum)
            print(ggnn_sum.size())
            result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
            print(result.size())
            return result


class MLP(nn.Module):
    def __init__(self, i_dim, o_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(i_dim, 100)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, o_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RelGraphConvModel(nn.Module):
    def __init__(self,
                 input_dim, h_dim, out_dim, num_relations,
                 num_bases=-1, num_hidden_layers=1, regularizer='basis',
                 dropout=0., self_loop=False, ns_mode=False):
        super(RelGraphConvModel, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.ns_mode = ns_mode

        if self.num_bases == -1:
            self.num_bases = self.num_relations

        self.h_layers = nn.ModuleList()
        self.in_layer = RelGraphConv(self.input_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        for _ in range(self.num_hidden_layers):
            self.h_layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop))
        self.out_layer = RelGraphConv(self.h_dim, self.out_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        self.readout = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))
        self.classifier = MLP(self.out_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, g, cuda=False, device=None, output_ggnn=False, output_attention=False):
        if self.ns_mode:
            # forward for neighbor sampling
            x = g[0].ndata['features']
            # input layer
            h = self.in_layer(g[0], x, g[0].edata['etype'])
            h = self.dropout(f.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(g[idx+1], h, g[idx+1].edata['etype'])
                h = self.dropout(f.relu(h))
            # output layer
            idx = len(self.h_layers) + 1
            h = self.out_layer(g[idx], h, g[idx].edata['etype'])
            return h
        else:
            x = g.ndata['features']
            e = g.edata['etype']
            if cuda:
                g = g.to(device)
                x = x.to(device)
                e = e.to(device)
            if output_ggnn:
                graph_list = dgl.unbatch(g)
                node_embeds = [graph.ndata['features'] for graph in graph_list]
            # input layer
            h = self.in_layer(g, x, e)
            h = self.dropout(f.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(g, h, e)
                h = self.dropout(f.relu(h))
            # output layer
            h = self.out_layer(g, h, e)
            if output_ggnn:
                tmp_hidden_states = h
                h = self.readout(g, h)  # shape (batch, out_dim)
                h = self.activation(self.classifier(h)).squeeze(dim=-1)  # shape (batch,)
                g.ndata['features'] = tmp_hidden_states
                graph_list = dgl.unbatch(g)
                graph_embeds = [graph.ndata['features'] for graph in graph_list]
                return h, node_embeds, graph_embeds
            else:
                # readout function
                if output_attention:
                    h, gate = self.readout(g, h, get_attention=True)  # shape (batch, out_dim)
                else:
                    h = self.readout(g, h)  # shape (batch, out_dim)
                h = self.activation(self.classifier(h)).squeeze(dim=-1)  # shape (batch,)
                if output_attention:
                    return h, gate
                else:
                    return h


class BertWithGAP(nn.Module):
    def __init__(self,
                 input_dim, h_dim, out_dim, num_relations,
                 encoder, tokenizer, config,
                 num_bases=-1, num_hidden_layers=1, regularizer='basis',
                 dropout=0., self_loop=False, ns_mode=False):
        super(BertWithGAP, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.config = config
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.ns_mode = ns_mode

        self.dense = nn.Linear(self.config.hidden_size, self.out_dim)

        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.num_bases == -1:
            self.num_bases = self.num_relations

        self.h_layers = nn.ModuleList()
        self.in_layer = RelGraphConv(self.input_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        for _ in range(self.num_hidden_layers):
            self.h_layers.append(RelGraphConv(self.h_dim, self.h_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop))
        self.out_layer = RelGraphConv(self.h_dim, self.out_dim, self.num_relations, self.regularizer,
                            self.num_bases, self_loop=self.self_loop)
        self.readout = GlobalAttentionPooling(nn.Linear(self.out_dim, 1))
        self.classifier = MLP(self.out_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, g, input_ids=None, input_embeds=None, node_ids_map=None, cuda=False, device=None, output_ggnn=False):
        # graph conv module
        x = g.ndata['features']
        e = g.edata['etype']
        if cuda:
            g = g.to(device)
            x = x.to(device)
            e = e.to(device)
        # input layer
        h = self.in_layer(g, x, e)
        h = self.dropout(f.relu(h))
        # hidden layers
        for idx, layer in enumerate(self.h_layers):
            h = layer(g, h, e)
            h = self.dropout(f.relu(h))
        # output layer
        h = self.out_layer(g, h, e)
        g.ndata['features'] = h
        unbatched_graph = dgl.unbatch(g)
        # ptm module
        if input_ids is not None:
            input_ids = input_ids.to(device)
            outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1))[0]
        else:
            input_ids = input_ids.to(device)
            outputs = self.encoder.roberta(inputs_embeds=input_embeds)[0]
        added_tensors = []
        for ubg, enc, imap in zip(unbatched_graph, outputs, node_ids_map):
            gf = ubg.ndata['features']
            enc = self.dense(enc)
            dense_enc = []
            for ids in imap:
                if ids:
                    selected = enc[ids, :]
                    if selected.size()[0] > 1:
                        selected = torch.mean(selected, dim=0, keepdims=True)
                else:
                    selected = torch.zeros(1, 200).to(device)
                dense_enc.append(selected)
            dense_enc = torch.stack(dense_enc).squeeze()
            gf = torch.add(gf, dense_enc)
            added_tensors.append(gf)
        h = torch.cat(added_tensors, dim=0)
        h = self.readout(g, h)  # shape (batch, out_dim)
        h = self.activation(self.classifier(h)).squeeze(dim=-1)  # shape (batch,)
        return h


class GatedMultiAttention(nn.Module):
    def __init__(self,
                 input_dim, h_dim, out_dim, num_relations,
                 num_bases=-1, num_hidden_layers=1, regularizer='basis',
                 dropout=0., self_loop=False, ns_mode=False):
        super(GatedMultiAttention, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.dropout = nn.Dropout(dropout)
        self.self_loop = self_loop
        self.ns_mode = ns_mode

        if self.num_bases == -1:
            self.num_bases = self.num_relations

        self.h_layers = nn.ModuleList()
        self.in_layer = GatedGraphConv(in_feats=self.input_dim, out_feats=self.h_dim, n_steps=6,
                                   n_etypes=1)
        for _ in range(self.num_hidden_layers):
            self.h_layers.append(GatedGraphConv(in_feats=self.h_dim, out_feats=self.h_dim, n_steps=6,
                                   n_etypes=1))
        self.out_layer = GatedGraphConv(in_feats=self.h_dim, out_feats=self.out_dim, n_steps=6,
                                   n_etypes=1)
        self.readout = GlobalAttentionPooling(nn.Linear(3 * self.out_dim, 1))
        self.classifier = MLP(3 * self.out_dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, g, cuda=False, device=None, output_ggnn=False):
        etypes = [1, 2, 3]
        subtensors = []
        for etype in etypes:
            etype_edges = (g.edata['etype'] == etype).nonzero().squeeze()
            subgraph = g.edge_subgraph(etype_edges, relabel_nodes=False)
            x = subgraph.ndata['features']
            e = subgraph.edata['etype']
            if cuda:
                subgraph = subgraph.to(device)
                x = x.to(device)
                e = e.to(device)
            # input layer
            h = self.in_layer(subgraph, x)
            h = self.dropout(f.relu(h))
            # hidden layers
            for idx, layer in enumerate(self.h_layers):
                h = layer(subgraph, h)
                h = self.dropout(f.relu(h))
            # output layer
            h = self.out_layer(subgraph, h)
            subtensors.append(h)
        outputs = torch.cat(subtensors, dim=-1)
        # readout function
        if cuda:
            g = g.to(device)
        outputs = self.readout(g, outputs)  # shape (batch, out_dim)
        outputs = self.activation(self.classifier(outputs)).squeeze(dim=-1)  # shape (batch,)
        return outputs
