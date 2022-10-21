import time
import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from models import *
import os
import torch
from torch_geometric.loader import DataLoader
import ogb.nodeproppred as node_eval
import ogb.graphproppred as graph_eval
import matplotlib.pyplot as plt

nb_train = 3
epoch = 100



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def write_results_to_file(task, nb_train,  best_epoch, best_acc, valid_acc, test_acc, total_time):
    valid_acc_str = [str(round(i, 2)) for i in valid_acc]
    f = open("result_"+task+".txt", "w")
    f.write("Number of training: " + str(nb_train) + "\n")
    f.write("Mean best valid accuracy: " + str(round(best_acc, 2)) + "% at mean epoch:" + str(best_epoch) + "\n")
    f.write("Mean test accuracy: " + str(round(test_acc, 2)) + "%" + "\n")
    f.write("Exec time: " + str(round(total_time, 2)) + " sec" + "\n")
    f.write("Last valid acc list: " + ' '.join(valid_acc_str) + "\n")
    f.close()


def print_results(best_epoch, best_acc, valid_acc, test_acc):
    valid_acc_str = [str(round(i, 2)) for i in valid_acc]
    print()
    print("Best valid accuracy: " + str(round(best_acc, 2)) + "% at epoch:", best_epoch)
    print("Test accuracy: " + str(round(test_acc, 2)) + "%")
    # print("Exec time: " + str(round(total_time, 2)) + " sec")
    print("Valid acc: " + ' '.join(valid_acc_str))

def plot_result(x):
    # Plotting the Graph
    plt.plot(x)
    plt.title("Valid accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()

def test(task, tloader):
    best_model = torch.load("models/best_model_"+task+".pth").to(device)
    acc = evaluate(best_model, tloader)
    return acc


def evaluate(model, loader):
    dataset = loader.dataset
    if model.task == 'node':
        evaluator = node_eval.Evaluator(name="ogbn-arxiv")
        with torch.no_grad():
            model.eval()
            out = model(data)
            y_pred = out.argmax(dim=-1, keepdim=True)
            acc = evaluator.eval({'y_true': data.y[dataset],
                                  'y_pred': y_pred[dataset],
                                  })['acc']
    else:
        evaluator = graph_eval.Evaluator(name="ogbg-molhiv")
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(loader):
                batch = batch.to(device)
                # y_true = torch.cat((y_true, batch.y.clone().detach().cpu()), dim=0)
                y_true.append(batch.y.clone().detach().cpu())
                out = model(batch)[:, 0]
                y_pred.append(out.clone().detach().cpu())
        y_true = torch.cat(y_true, dim=0)       # merci à Nathanael pour l idee de la liste concatenee
        y_pred = torch.cat(y_pred, dim=0).unsqueeze(1)
        acc = evaluator.eval({'y_true': y_true,
                              'y_pred': y_pred,
                              })['rocauc']
    return acc * 100


def train(task, epoch, data, len_data, tloader, vloader):
    if task == "node":
        model = NodeModel(max(dataset.num_node_features, 1), 128, dataset.num_classes, task).to(device)
    else:
        model = GraphModel(max(dataset.num_node_features, 1), 128, dataset.num_classes, task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    valid_acc = []
    best_acc = 0
    best_epoch = -1
    pbar = tqdm.trange(epoch)
    for epoch in pbar:
        epoch_loss = 0
        model.train()
        epoch += 1
        for idx, batch in enumerate(tloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            if model.task == 'graph':
                out = model(batch)
                y = torch.flatten(batch.y)
                loss = model.loss(out, y)
            else:
                out = model(data)
                loss = model.loss(out[batch], torch.flatten(data.y[batch]))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, vloader)
        valid_acc.append(acc)
        epoch_loss = epoch_loss / len_data['train']
        bar_str = "train loss="+f"{epoch_loss:.2E}"+"; valid acc="+str(round(valid_acc[-1], 2))+"%"
        pbar.set_postfix_str(bar_str)
        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            try:
                torch.save(model, "models/best_model_"+model.task+".pth")
            except:
                time.sleep(0.5)
                torch.save(model, "models/best_model_" + model.task + ".pth")
    return best_epoch, best_acc, valid_acc


task = "graph"

if not os.path.exists("models"):
    os.mkdir("models")

if task == "node":
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=torch_geometric.transforms.ToSparseTensor()) # https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.ToSparseTensor
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(split_idx["train"], batch_size=len(split_idx["train"]), shuffle=True)
    # train_loader = DataLoader(split_idx["train"], batch_size=512, shuffle=True)
    valid_loader = DataLoader(split_idx["valid"], batch_size=len(split_idx["valid"]), shuffle=False)
    test_loader = DataLoader(split_idx["test"], batch_size=len(split_idx["test"]), shuffle=False)

elif task == "graph":
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    data = dataset[0].to(device)

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=256, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=len(split_idx["valid"]), shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=len(split_idx["test"]), shuffle=False)

print(dataset)

m_test_acc = 0
m_valid_acc = 0
m_best_epoch = 0
start_time = time.time()
for i in range(nb_train):
    best_epoch, best_acc, valid_acc = train(task, epoch, data,  {"train": len(split_idx["train"])}, train_loader, valid_loader)
    if nb_train == 1:
        plot_result(valid_acc)
    test_acc = test(task, test_loader)
    print_results(best_epoch, best_acc, valid_acc, test_acc)
    m_test_acc += test_acc
    m_valid_acc += best_acc
    m_best_epoch += best_epoch
m_best_epoch = m_best_epoch / nb_train
m_test_acc = m_test_acc / nb_train
m_valid_acc = m_valid_acc / nb_train
total_time = time.time() - start_time
write_results_to_file(task, nb_train, m_best_epoch, m_valid_acc, valid_acc, m_test_acc, total_time/nb_train)

