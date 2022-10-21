import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator as Node_evaluaor
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator as Graph_evaluator

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from model import GCN, GIN, GAT, GEN, SAGE
from model import GCN_Graph
from gtrick import FLAG

import argparse
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def node_property_prediction(flag: bool):
    
    num_layers = 3
    hidden_dim = 128
    dropout = 0.5
    lr = 0.001
    epochs = 600


    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name,
                                    transform=T.ToSparseTensor())
    data = dataset[0]

    # Check task type
    print('Task type: {}'.format(dataset.task_type))

    # Make the adjacency matrix to symmetric
    data.adj_t = data.adj_t.to_symmetric()

    # If you use GPU, the device should be cuda
    print('Device: {}'.format(device))

    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = SAGE(data.num_features, hidden_dim,
                dataset.num_classes, num_layers,
                dropout).to(device)
    evaluator = Node_evaluaor(name='ogbn-arxiv')

    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = F.nll_loss
    # define flag, params: in_feats, loss_func, optimizer
    if flag :
        flag_obj = FLAG(data.x.shape[1], loss_fn, optimizer)
    else :
        flag_obj = None
    best_valid_acc = 0

    for epoch in range(1, 1 + epochs):
        loss = train_node_property_prediction(model, data, train_idx, optimizer, loss_fn, flag_obj)
        result = test_node_property_prediction(model, data, split_idx, evaluator)
        train_acc, valid_acc, test_acc = result
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%')

    best_result = test_node_property_prediction(best_model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = best_result
    print(f'Best model: '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')




def train_node_property_prediction(model, data, train_idx, optimizer, loss_fn, flag):
    
    model.train()
    loss = 0
    
    optimizer.zero_grad()

    if flag :
        # Feed the data into the model
        y_ = model(data.x, data.adj_t)
        y = data.y[train_idx]

        # define a forward func to get the output of the model
        forward = lambda perturb: model(data.x, data.adj_t, perturb)[train_idx]

        # run flag to get loss and output
        loss, out = flag(model, forward, data.x.shape[0], y.squeeze(1))


    else :
       # Feed the data into the model
        y_ = model(data.x, data.adj_t)
        y = data.y
        loss = loss_fn(y_[train_idx], y[train_idx, 0])

        loss.backward()
        optimizer.step()
    
    return loss.item()



# Test function here
@torch.no_grad()
def test_node_property_prediction(model, data, split_idx, evaluator):
    model.eval()    

    # No index slicing here
    out = model(data.x, data.adj_t)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc



def graph_property_prediction():

    num_layers = 2
    hidden_dim = 256
    dropout = 0.5
    lr = 0.001
    epochs = 70
    gnn_type = 'sage'

    # Load the dataset 
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

    print('Device: {}'.format(device))

    split_idx = dataset.get_idx_split()

    # Check task type
    print('Task type: {}'.format(dataset.task_type))

    # Load the data sets into dataloader
    # We will train the graph classification task on a batch of 64 graphs
    # Shuffle the order of graphs for training set
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=64, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=64, shuffle=False, num_workers=0)

    model = GCN_Graph(hidden_dim,
            dataset.num_tasks, num_layers,
            dropout, type=gnn_type).to(device)
    evaluator = Graph_evaluator(name='ogbg-molhiv')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_valid_acc = 0
    for epoch in range(1, 1 + epochs):
        loss = train_graph_property_prediction(model, device, train_loader, optimizer, loss_fn)
        train_result = eval_graph_property_prediction(model, device, train_loader, evaluator)
        val_result = eval_graph_property_prediction(model, device, valid_loader, evaluator)

        valid_acc = val_result[dataset.eval_metric]
        train_acc = train_result[dataset.eval_metric]
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(model)
        print(f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}% '
                f'Valid: {100 * valid_acc:.2f}% ')
    
    train_acc = eval_graph_property_prediction(best_model, device, train_loader, evaluator)[dataset.eval_metric]
    valid_acc = eval_graph_property_prediction(best_model, device, valid_loader, evaluator)[dataset.eval_metric]
    test_acc = eval_graph_property_prediction(best_model, device, test_loader, evaluator)[dataset.eval_metric]

    print(f'Best model: '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')

def train_graph_property_prediction(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0
    
    for step, batch in enumerate(tqdm(data_loader, desc="Trainning")):
      batch = batch.to(device)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
          pass
      else:
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        # `is_labeled` mask filter output and labels
        optimizer.zero_grad()
        
        y_ = model(batch)[is_labeled].view(-1)
        y = batch.y[is_labeled].view(-1).float()
        loss = loss_fn(y_, y)

        loss.backward()
        optimizer.step()

    return loss.item()

# The evaluation function
def eval_graph_property_prediction(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Evaluation")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def main():
    parser = argparse.ArgumentParser(
        description='train node property prediction or train graph property prediction')
    parser.add_argument('dataset', type=str, choices=['ogbn-arxiv', 'ogbg-molhiv'])
    parser.add_argument('--flag', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    if args.dataset == 'ogbg-molhiv' :
        graph_property_prediction()
    elif args.dataset == 'ogbn-arxiv' :
        node_property_prediction(args.flag)
    else :
        raise ValueError(f'wrong dataset {args.dataset}')
if __name__ == '__main__':
    main()