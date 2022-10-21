import os
modelSavePath = os.path.dirname(__file__) + "/models"
os.makedirs(modelSavePath, exist_ok=True)

import torch

import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, ChebConv, GATConv, SAGEConv, Linear, GINConv


class SuperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def save(self, name):
        torch.save(self.state_dict(), f"{modelSavePath}/{name}.torch")

    def load(self, name):

        self.load_state_dict(torch.load(f"{modelSavePath}/{name}.torch"))


class GCN2BN1LIN2MLP3(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.features_dim = parameters["features_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.batch_norm = torch_geometric.nn.BatchNorm(self.features_dim)
        self.conv1 = GCNConv(self.features_dim, self.hidden_dim)
        self.lin1 = Linear(self.hidden_dim, self.hidden_dim)

        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.lin2 = Linear(self.hidden_dim, self.hidden_dim)

        self.mpl1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mpl2 = Linear(self.hidden_dim, self.hidden_dim)
        self.mpl3 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.batch_norm(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        #x = self.mlp1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        #x = self.mlp2(x)

        x = self.mpl1(x)
        x = self.mpl2(x)
        x = self.mpl3(x)

        return F.log_softmax(x, dim=1)


class GCN2LIN2MLP3(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.features_dim = parameters["features_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.conv1 = GCNConv(self.features_dim, self.hidden_dim)
        self.lin1 = Linear(self.hidden_dim, self.hidden_dim)

        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.lin2 = Linear(self.hidden_dim, self.hidden_dim)

        self.mpl1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mpl2 = Linear(self.hidden_dim, self.hidden_dim)
        self.mpl3 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        #x = self.mlp1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        #x = self.mlp2(x)

        x = self.mpl1(x)
        x = self.mpl2(x)
        x = self.mpl3(x)

        return F.log_softmax(x, dim=1)
