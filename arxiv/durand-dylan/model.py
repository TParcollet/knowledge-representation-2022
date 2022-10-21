import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

from torch_geometric.nn import GCNConv, GCN2Conv, GraphConv, SAGEConv, GINConv, GATv2Conv, GENConv
from torch_geometric.nn import GraphNorm, InstanceNorm, LayerNorm
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, type ="gcn"):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim

        if type == "gcn" :
            self.gnn_node = GCN(hidden_dim, hidden_dim,
                hidden_dim, num_layers, dropout, return_embeds=True)
        elif type == "gen" :
            self.gnn_node = GEN(hidden_dim, hidden_dim,
                hidden_dim, num_layers, dropout, return_embeds=True)
        elif type == "sage" :
            self.gnn_node = SAGE(hidden_dim, hidden_dim,
                hidden_dim, num_layers, dropout, return_embeds=True)
        elif type == "gin" :
            self.gnn_node = GIN(hidden_dim, hidden_dim,
                hidden_dim, num_layers, dropout, return_embeds=True)
        elif type == "gat" :
            self.gnn_node = GAT(hidden_dim, hidden_dim,
                hidden_dim, num_layers, dropout, return_embeds=True)
        else :
            raise ValueError(f"Wrong gnn type : {type}")


        self.pool = global_mean_pool

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        x = self.gnn_node(embed, edge_index)
        x = self.pool(x, batch)
        out = self.linear(x)

        return out



class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
            
        self.convs.append(GENConv(input_dim, output_dim))

        # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, perturb=None):
        # add perturb to x, note that do not use x += perturb
        if perturb is not None:
            x = x + perturb
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)
        return out


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GIN, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINConv(
                Sequential(Linear(input_dim, hidden_dim),
                        BatchNorm1d(hidden_dim), ReLU(),
                        Linear(hidden_dim, hidden_dim), ReLU())))
            input_dim = hidden_dim
            
        self.convs.append(GINConv(
                Sequential(Linear(hidden_dim, hidden_dim),
                        BatchNorm1d(hidden_dim), ReLU(),
                        Linear(hidden_dim, hidden_dim), ReLU())))

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)
        return out

# Graph attention network
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False, heads = 1):

        super(GAT, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GATv2Conv(input_dim, hidden_dim, heads))
            input_dim = hidden_dim
            
        self.convs.append(GATv2Conv(input_dim, output_dim, heads))

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)
        return out

# DeeperGCN
class GEN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(GEN, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GENConv(input_dim, hidden_dim))
            input_dim = hidden_dim
            
        self.convs.append(GENConv(input_dim, output_dim))

        # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, perturb=None):
        # add perturb to x, note that do not use x += perturb
        if perturb is not None:
            x = x + perturb
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)
        return out

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):

        super(SAGE, self).__init__()

        # A list of GCNConv layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            input_dim = hidden_dim
            
        self.convs.append(SAGEConv(input_dim, output_dim))

        # A list of 1D batch normalization layers
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        # The log softmax layer
        self.softmax = torch.nn.LogSoftmax()

        # Probability of an element to be zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, perturb=None):
        # add perturb to x, note that do not use x += perturb
        if perturb is not None:
            x = x + perturb
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        if self.return_embeds:
            out = x
        else:
            out = self.softmax(x)
        return out