import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as geo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

# Je me suis aidé de ce lien pour atteindre les 70% sinon je bloquais à 50%:
# https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=2ccbce231c5517c65c5bf296aefb099bd63f90a8&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f786965636b31332f6373323234772d77696e7465722d323032312f326363626365323331633535313763363563356266323936616566623039396264363366393061382f4353323234575f436f6c61625f322e6970796e62&logged_in=false&nwo=xieck13%2Fcs224w-winter-2021&path=CS224W_Colab_2.ipynb&platform=android&repository_id=356801771&repository_type=Repository&version=99
# grace à ce lien j ai utilisé : "data.adj_t = data.adj_t.to_symmetric()"

class NodeModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task="node"):
        super(NodeModel, self).__init__()
        self.task = task
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Tâche inconnue :-(')
        self.dropout = 0.25
        self.num_layers = 3
        self.convs = torch.nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        for i in range(self.num_layers - 1):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.norms = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.norms.append(torch.nn.BatchNorm1d(hidden_dim))
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Dropout(0.25),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(hidden_dim, output_dim))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, output_dim))

    def build_conv_model(self, input_dim, hidden_dim):
        return torch_geometric.nn.GCNConv(input_dim, hidden_dim)
        # return torch_geometric.nn.GINConv(torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim)))
        # return torch_geometric.nn.SAGEConv(input_dim, hidden_dim)

    def loss(self, x, y):
        return F.nll_loss(x, y)

    def forward(self, data):
        x, batch, adj_t = data.x, data.batch, data.adj_t
        for i in range(self.num_layers-1):
            x = self.convs[i](x, adj_t)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norms[-1](x)
        x = self.mlp(x)
        return F.log_softmax(x, dim=1)

#
# class GraphModel(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, task="node"):
#         super(GraphModel, self).__init__()
#         self.task = task
#         if not (self.task == 'node' or self.task == 'graph'):
#             raise RuntimeError('Tâche inconnue :-(')
#         self.dropout = 0.25
#         self.num_layers = 2
#         self.data_norm = torch_geometric.nn.BatchNorm(input_dim)
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(self.build_conv_model(100, hidden_dim))
#         self.encoder = AtomEncoder(emb_dim=100)
#         # Notre profondeur est de 2 (i.e 2 sauts dans le voisinage)
#         for l in range(self.num_layers-1):
#             self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
#         self.conv_norm = torch_geometric.nn.BatchNorm(hidden_dim)
#         # Une fois ces sauts effectués et le message passé,
#         # nous appliquons quelques non-linéaritées.
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.Dropout(0.25),
#             torch.nn.ReLU(),
#             torch.nn.Linear(hidden_dim, output_dim))
#
#
#     def build_conv_model(self, input_dim, hidden_dim):
#         # return torch_geometric.nn.GATv2Conv(input_dim, hidden_dim)
#         # return torch_geometric.nn.GCNConv(input_dim, hidden_dim)
#         return torch_geometric.nn.SAGEConv(input_dim, hidden_dim)
#         # return torch_geometric.nn.GINConv(torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim)))
#
#     def loss(self, x, y):
#         return F.nll_loss(x, y)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         # x = x.type(torch.float32)
#         x = self.encoder(x)
#         if data.num_node_features == 0:
#             x = torch.ones(data.num_nodes, 1)
#         for i in range(self.num_layers):
#             x = self.convs[i](x, edge_index)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.conv_norm(x)
#         x = torch_geometric.nn.global_mean_pool(x, batch)
#
#         x = self.mlp(x)
#         return F.log_softmax(x, dim=1)