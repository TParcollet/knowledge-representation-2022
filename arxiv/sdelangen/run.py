import sys
from argparse import ArgumentParser

import ogb.graphproppred
import ogb.nodeproppred
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm

# This "glue" code was largely inspired from PyTorch Geometric examples:
# https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/main_pyg.py


def train_graphprop(model, device, loader, scaler, optimizer):
    model.train()

    total_loss = torch.tensor(0.0).to(device)

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        model.zero_grad(set_to_none=True)

        with torch.autocast(device):
            emb, pred = model(batch)
            label = batch.y
            loss = model.loss(pred, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.detach() * batch.num_graphs

    total_loss /= len(loader.dataset)

    return total_loss.item()


def train_nodeprop(
    model, device, input_data, adj_t, target_data, split, scaler, optimizer
):
    model.train()

    model.zero_grad(set_to_none=True)

    with torch.autocast(device):
        pred = model(input_data, adj_t)
        pred = pred[split]
        label = target_data

        loss = model.loss(pred, label)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()


def eval_graphprop(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            emb, pred = model(batch)

        label = batch.y
        y_true.append(label.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})


def eval_nodeprop(model, device, input_data, adj_t, target_data, split, evaluator):
    model.eval()

    with torch.no_grad(), torch.autocast(device):
        pred = model(input_data, adj_t)
        pred = torch.argmax(pred[split], dim=-1, keepdim=True)

    y_true = target_data.view(pred.shape, -1).detach().cpu()
    y_pred = pred.detach().cpu()
    assert y_true.shape == y_pred.shape

    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("comment")

    # ogbg-molhiv:
    # Embedding size of 64 seems like the sweet spot. 32 results in mediocre
    # performance for most architectures I've tried.
    # ogbn-arxiv:
    # Embedding size of 64 seems to perform reasonably well in general.
    # Higher values were annoying for testing due to OOMs.
    # atom-emb-size is unused on arxiv
    parser.add_argument("--atom-emb-size", type=int, default=64)
    parser.add_argument("--emb-size", type=int, default=64)

    # With an embedding size of 64, a depth of 3 or 4 seems to maximize test
    # accuracy. higher worsens accuracy compared to depth==1
    # Increasing the embedding size with higher depth (==4) worsened accuracy.
    parser.add_argument("--gnn-depth", type=int, default=2)

    args = parser.parse_args()

    d_name = args.dataset

    if args.task == "graph":
        dataset = ogb.graphproppred.PygGraphPropPredDataset(name=d_name)
        evaluator = ogb.graphproppred.Evaluator(name=d_name)

        def make_loader(split_name, is_train):
            return DataLoader(
                dataset[split_idx[split_name]],
                batch_size=2048,  # big, but works well with batchnorm
                shuffle=is_train,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True,  # yolo better across epochs
                prefetch_factor=30,  # yolo
            )

        split_idx = dataset.get_idx_split()
        train_loader = make_loader("train", True)
        valid_loader = make_loader("valid", False)
        test_loader = make_loader("test", False)
    else:
        dataset = ogb.nodeproppred.PygNodePropPredDataset(
            name=d_name, transform=T.ToSparseTensor()
        )
        evaluator = ogb.nodeproppred.Evaluator(name=d_name)
        split_idx = dataset.get_idx_split()

    print(
        f"Training for dataset {d_name}, task is {dataset.task_type}", file=sys.stderr
    )

    runs_dir = f"runs-{d_name}/{args.comment}"
    writer = SummaryWriter(log_dir=runs_dir, comment=args.comment)

    torch.backends.cudnn.benchmark = True

    if d_name == "ogbg-molhiv":
        from molhivgnn import GNN

        model = GNN(
            atom_emb_dim=args.atom_emb_size,
            hidden_dim=args.emb_size,
            gnn_depth=args.gnn_depth,
            output_dim=1,
        ).to(args.device)
    elif d_name == "ogbn-arxiv":
        from arxivgnn import GNN

        model = GNN(
            hidden_dim=args.emb_size, gnn_depth=args.gnn_depth, output_dim=40
        ).to(args.device)
    else:
        raise ValueError("unknown dataset specified")

    # Observations for the ogbg-molhiv dataset:
    #   A LR of 0.001 worked, but I suspected it to be too high and to cause
    #   variance between trainings.
    #   0.0003 as an attempt to mitigate this converged too slowly.
    #   0.0007 did not help.
    #   Experimenting with a _higher_ LR of 0.003 resulted in faster convergence and similar results.
    #   Variance between trainings likely cannot be attributed to the optimizer here.
    #   Rather, it may be because the validation set is too small.
    #   0.005 and a raised epoch count of 200 showed that the model could still learn.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Observations for the obgb-molhiv dataset:
    #   The cosine annealing LR schedule was recommended online as potentially helpful for AdamW.
    #   In practice, it didn't seem to make much of a difference to either convergence speed or final accuracy.
    #   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    #
    #   Linearly decaying the LR from 0.005*1 to 0.005*0.05 did not seem to have the intended "fine tuning" effect
    #   on the long run, even over a high number of epochs.
    #   scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=args.epochs)
    #
    #   This seems to prevent any meaningful improvement when training stalls for a couple of epochs
    #   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    #
    #   Improve training times with AMP with gradient scaling.
    scaler = torch.cuda.amp.GradScaler()

    best_valid_score = 0
    best_test_score = 0

    if args.task == "node":
        input_data = dataset[0].x.to(args.device)
        target = dataset[0].y.squeeze(1).to(args.device)
        adj_t = dataset[0].adj_t.to_symmetric().to(args.device)

    if args.task == "graph":
        metric = "rocauc"
    else:
        metric = "acc"

    for epoch in tqdm(range(1, args.epochs + 1)):
        if args.task == "graph":
            train_loss = train_graphprop(
                model, args.device, train_loader, scaler, optimizer
            )
        else:
            train_loss = train_nodeprop(
                model,
                args.device,
                input_data,
                adj_t,
                target[split_idx["train"]],
                split_idx["train"],
                scaler,
                optimizer,
            )

        # scheduler.step(train_loss) # for ReduceLROnPlateau

        if args.task == "graph":
            current_valid_score = eval_graphprop(
                model, args.device, valid_loader, evaluator
            )
        else:
            current_valid_score = eval_nodeprop(
                model,
                args.device,
                input_data,
                adj_t,
                target[split_idx["valid"]],
                split_idx["valid"],
                evaluator,
            )

        if current_valid_score[metric] > best_valid_score:
            best_valid_score = current_valid_score[metric]

            if args.task == "graph":
                best_test_score = eval_graphprop(
                    model, args.device, test_loader, evaluator
                )[metric]
            else:
                best_test_score = eval_nodeprop(
                    model,
                    args.device,
                    input_data,
                    adj_t,
                    target[split_idx["test"]],
                    split_idx["test"],
                    evaluator,
                )[metric]

            writer.add_scalar("Accuracy/test", best_test_score, epoch)

        writer.add_scalar("Accuracy/valid", current_valid_score[metric], epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)

    print(best_test_score)


if __name__ == "__main__":
    main()
