import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.utils import add_remaining_self_loops


class CustomPassing(pyg_nn.MessagePassing):
    def __init__(self, emb_dim, output_dim):
        # keeping the attn related transforms simple seems to give similar perf
        compute_attn = nn.Sequential(
            nn.Linear(emb_dim, 1),
        )

        transform_node_feats = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
        )

        # attentionalaggregation slightly outperforms mean, but not by much
        aggr = pyg_nn.aggr.AttentionalAggregation(compute_attn, transform_node_feats)

        # super(CustomPassing, self).__init__(
        #     aggr=pyg_nn.aggr.MultiAggregation([aggr, "mean", "std"])
        # )
        
        super(CustomPassing, self).__init__(aggr=aggr)

        self.transform = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim), # batchnorm applied here greatly speeds up convergence
            nn.Hardswish(),
            nn.Linear(emb_dim, output_dim),
        )

        self.emb_dim = emb_dim

    def forward(self, x, edge_index):
        edge_index, _edge_emb = add_remaining_self_loops(
            edge_index, num_nodes=x.size(0)
        )
        x = self.propagate(edge_index, x=x)
        x = self.transform(x)

        return x

    def message(self, x_j):
        return x_j

    # def message_and_aggregate(self, adj_t, x):
    #     return torch.matmul(adj_t, x, reduce=self.aggr)

    def update(self, aggr_out):
        # aggr_out: [N, out_channels]
        return aggr_out


class GNN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, gnn_depth):
        super(GNN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(128, hidden_dim))

        for i in range(gnn_depth):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        self.graph_norm = pyg_nn.GraphNorm(hidden_dim)

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.Hardswish(),
            nn.Linear(hidden_dim, output_dim),

            nn.LogSoftmax(-1)
        )

        self.num_layers = gnn_depth + 1
        self.dropout = 0.1

    def build_conv_model(self, input_dim, output_dim):
        return CustomPassing(input_dim, output_dim)

    def forward(self, x, adj_t):
        row, col, _edge_attr = adj_t.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        
        edge_index, _edge_attr = pyg_utils.dropout_adj(edge_index, p=0.1, training=self.training)

        # Propagate through the convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.hardswish(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return nn.NLLLoss()(pred, label)
