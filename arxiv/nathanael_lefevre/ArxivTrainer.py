from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator

import torch
import torch.nn.functional as F
import tqdm
import time

import matplotlib.pyplot as plt

import ArxivModels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")

class Trainer:
    def __init__(self, model, dataset, parameters):
        self.n_epoch = parameters["n_epoch"]
        self.batch_size = parameters["batch_size"]
        self.features_dim = parameters["features_dim"]
        self.learning_rate = parameters["learning_rate"]

        self.model = model
        self.data = dataset[0].to(device)
        split_idx = dataset.get_idx_split()
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        self.loaders = {"train": torch.utils.data.DataLoader(self.train_idx, batch_size=self.batch_size, shuffle=True),
                         "valid": torch.utils.data.DataLoader(self.valid_idx, batch_size=self.batch_size, shuffle=True),
                         "test": torch.utils.data.DataLoader(self.test_idx, batch_size=self.batch_size, shuffle=True)}
        self.evaluator = Evaluator(name="ogbn-arxiv")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        self.loss_func = F.nll_loss

        self.metrics = {}

    def train(self):
        time_start = time.time()
        self.metrics["best_valid_accuracy"] = 0
        self.metrics["duration"] = []
        self.metrics["loss"] = []
        self.model.to(device)

        tbar = tqdm.trange(self.n_epoch)
        tbar_desc = ""
        for epoch in tbar:
            self.model.train()
            total_loss, batch_mean_loss = .0, .0
            for batch, batch_idx in enumerate(self.loaders["train"]):
                tbar.set_description(f'Batch{batch}/{len(self.loaders["train"])}. ' + tbar_desc)
                self.optimizer.zero_grad()

                out = self.model(self.data)
                loss = self.loss_func(out[batch_idx], torch.flatten(self.data.y[batch_idx]))
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            batch_mean_loss = total_loss / len(self.loaders["train"])
            self.eval(mode="valid")
            self.metrics["duration"].append(time.time() - time_start)
            self.metrics["loss"].append(batch_mean_loss)

            self.pocket(epoch)
            tbar_desc = f'Current Valid Accuracy: {self.metrics["valid_accuracy"][-1]:.4f}, ' \
                        f'Best Accuracy: {self.metrics["best_valid_accuracy"]:.4f}, ' \
                        f'Batch Mean loss: {batch_mean_loss:.4f}'

        self.model.load("best_valid")
        self.eval(mode="test")

        return self.metrics

    def eval(self, mode):
        '''
        :param model: modèle d'ia
        :param loader: data loader
        :param mode: "valid" ou "test"
        '''
        with torch.no_grad():
            # Le système de liste y_true et y_pred a été inspiré par Eddy
            self.model.eval()
            y_true = []
            y_pred = []
            for batch in self.loaders[mode]:
                y_true.append(self.data.y[self.valid_idx])

                out = self.model(self.data)
                y_pred.append(torch.unsqueeze(out.argmax(dim=1)[self.valid_idx], dim=1).detach().cpu())

            y_true = torch.cat(y_true, dim=0).cpu().numpy()
            y_pred = torch.cat(y_pred, dim=0).numpy()

            acc = self.evaluator.eval({'y_true': y_true,
                                  'y_pred': y_pred})["acc"]

            if mode == "valid":
                # noinspection PyBroadException
                try:
                    self.metrics["valid_accuracy"].append(acc)
                except Exception as e:
                    self.metrics["valid_accuracy"] = [acc]
            else:
                self.metrics[f"{mode}_accuracy"] = acc

    def pocket(self, epoch):
        is_best_defined = "best_valid_accuracy" in self.metrics.keys()
        if (not is_best_defined) or self.metrics["valid_accuracy"][-1] > self.metrics["best_valid_accuracy"]:
            self.metrics["best_valid_accuracy"] = self.metrics["valid_accuracy"][-1]
            self.metrics["best_valid_epoch"] = epoch
            self.model.save("best_valid")


def main():
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")

    parameters = {"features_dim": dataset.num_node_features,
                  "n_classes": int(dataset.num_classes),
                  "hidden_dim": 256,
                  "dropout": .2,
                  "batch_size": 256,
                  "learning_rate": 0.001,
                  "n_epoch": 20}

    model = ArxivModels.GCN2LIN2MLP3(parameters=parameters).to(device)

    trainer = Trainer(model, dataset, parameters=parameters)
    metrics = trainer.train()

    print(metrics)

    for key in metrics.keys():
        # noinspection PyBroadException
        try:
            len(metrics[key])
            plt.plot(metrics[key])
            plt.title(key)
            plt.show()
            plt.close()
        except Exception as e:
            pass


if __name__ == "__main__":
    main()
