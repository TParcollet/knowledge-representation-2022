import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_remaining_self_loops


class CustomGINConv(pyg_nn.MessagePassing):
    def __init__(self, emb_dim, output_dim):
        # keeping the attn related transforms simple seems to give similar perf
        compute_attn = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )

        transform_node_feats = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
        )

        aggr = pyg_nn.aggr.AttentionalAggregation(compute_attn, transform_node_feats)

        super(CustomGINConv, self).__init__(
            aggr=pyg_nn.aggr.MultiAggregation(["mean", "std", aggr])
        )
        # super(CustomGINConv, self).__init__(aggr=aggr)

        self.transform = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Hardswish(),
            nn.Linear(emb_dim, output_dim),
        )

        self.bond_encoder = BondEncoder(emb_dim=output_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.bond_encoder(edge_attr)
        edge_index, edge_emb = add_remaining_self_loops(
            edge_index, num_nodes=x.size(0), edge_attr=edge_emb
        )
        x = self.transform(self.propagate(edge_index, x=x, edge_attr=edge_emb))
        return x

    def message(self, x_j, edge_attr):
        return F.hardswish(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, atom_emb_dim, hidden_dim, output_dim, gnn_depth):
        super(GNN, self).__init__()

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)

        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(atom_emb_dim, hidden_dim))

        for i in range(gnn_depth):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.25),
            nn.Hardswish(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.num_layers = gnn_depth + 1
        self.dropout = 0.25

    def build_conv_model(self, input_dim, output_dim):
        return CustomGINConv(input_dim, output_dim)

    def forward(self, data):
        x = self.atom_encoder(data.x)

        edge_index, edge_attr = data.edge_index, data.edge_attr
        edge_index, edge_attr = pyg_utils.dropout_adj(edge_index, edge_attr, p=0.1, training=self.training)

        # Propagate through the convolution layers
        for i in range(self.num_layers):
            # TODO: use edge attributes: data.edge_attr; MessagePassing child may be required
            x = self.convs[i](x, edge_index, edge_attr)
            emb = x
            x = F.hardswish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # add/mean/max pooling seems to yield similar results
        x = pyg_nn.global_mean_pool(x, data.batch)

        x = self.post_mp(x)

        return emb, x

    def loss(self, pred, label):
        return F.binary_cross_entropy_with_logits(pred, label.float())
