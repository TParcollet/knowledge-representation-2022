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


class SAGE3BN3MLP2_noGlobalMeanPool(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.emb_dim = parameters["emb_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.conv1 = SAGEConv(self.emb_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = SAGEConv(self.hidden_dim, self.hidden_dim)

        self.batch_norm1 = torch_geometric.nn.BatchNorm(self.hidden_dim)
        self.batch_norm2 = torch_geometric.nn.BatchNorm(self.hidden_dim)
        self.batch_norm3 = torch_geometric.nn.BatchNorm(self.hidden_dim)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mlp2 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.mlp1(x)
        x = self.mlp2(x)

        return F.log_softmax(x, dim=1)


class SAGE3BN3MLP2(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.emb_dim = parameters["emb_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.conv1 = SAGEConv(self.emb_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = SAGEConv(self.hidden_dim, self.hidden_dim)

        self.batch_norm1 = torch_geometric.nn.BatchNorm(self.hidden_dim)
        self.batch_norm2 = torch_geometric.nn.BatchNorm(self.hidden_dim)
        self.batch_norm3 = torch_geometric.nn.BatchNorm(self.hidden_dim)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mlp2 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.mlp1(x)
        x = self.mlp2(x)

        return F.log_softmax(x, dim=1)


class SAGE3BN1MLP2(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.emb_dim = parameters["emb_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.batch_norm = torch_geometric.nn.BatchNorm(self.emb_dim)

        self.conv1 = SAGEConv(self.emb_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = SAGEConv(self.hidden_dim, self.hidden_dim)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mlp2 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, edge_index, batch):
        x = self.batch_norm(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.mlp1(x)
        x = self.mlp2(x)

        return F.log_softmax(x, dim=1)


class GCN3BN3MLP2(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.emb_dim = parameters["emb_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.conv1 = GCNConv(self.emb_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.batch_norm1 = torch_geometric.nn.BatchNorm(self.hidden_dim)
        self.batch_norm2 = torch_geometric.nn.BatchNorm(self.hidden_dim)
        self.batch_norm3 = torch_geometric.nn.BatchNorm(self.hidden_dim)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mlp2 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv3(x, edge_index)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.mlp1(x)
        x = self.mlp2(x)

        return F.log_softmax(x, dim=1)


class GCN3BN1MLP2(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.emb_dim = parameters["emb_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.batch_norm = torch_geometric.nn.BatchNorm(self.emb_dim)

        self.conv1 = GCNConv(self.emb_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mlp2 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, edge_index, batch):
        x = self.batch_norm(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.mlp1(x)
        x = self.mlp2(x)

        return F.log_softmax(x, dim=1)


class GCN2MLP2(SuperModel):
    def __init__(self, parameters):
        super().__init__()
        self.emb_dim = parameters["emb_dim"]
        self.n_classes = parameters["n_classes"]
        self.hidden_dim = parameters["hidden_dim"]
        self.dropout = parameters["dropout"]

        self.conv1 = GCNConv(self.emb_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.mlp1 = Linear(self.hidden_dim, self.hidden_dim)
        self.mlp2 = Linear(self.hidden_dim, self.n_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)

        x = torch_geometric.nn.global_mean_pool(x, batch)
        x = self.mlp1(x)
        x = self.mlp2(x)

        return F.log_softmax(x, dim=1)