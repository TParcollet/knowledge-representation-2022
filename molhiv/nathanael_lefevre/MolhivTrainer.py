from ogb.graphproppred import PygGraphPropPredDataset
import torch
import torch.nn.functional as F
import json
import tqdm
import time

from ogb.graphproppred.mol_encoder import AtomEncoder  # , BondEncoder
from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

import MolhivModels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")


class Trainer:
    def __init__(self, model, dataset, parameters):
        self.n_epoch = parameters["n_epoch"]
        self.batch_size = parameters["batch_size"]
        self.emb_dim = parameters["emb_dim"]
        self.learning_rate = parameters["learning_rate"]

        self.model = model

        dataset.data.to(device)

        split_idx = dataset.get_idx_split()
        self.loaders = {"train": DataLoader(dataset[split_idx["train"]],
                                            batch_size=self.batch_size,
                                            shuffle=True),
                        "valid": DataLoader(dataset[split_idx["valid"]],
                                            batch_size=len(split_idx["valid"]),
                                            shuffle=False),
                        "test": DataLoader(dataset[split_idx["test"]],
                                           batch_size=len(split_idx["test"]),
                                           shuffle=False)}

        self.evaluator = Evaluator(name="ogbg-molhiv")

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        self.encoder = AtomEncoder(emb_dim=self.emb_dim)  # BondEncoder(emb_dim=100)
        self.loss_func = F.cross_entropy  # F.nll_loss

        self.metrics = {}

    def train(self):
        time_start = time.time()
        self.metrics["best_valid_rocauc"] = 0
        self.metrics["duration"] = []
        self.metrics["loss"] = []
        self.model.to(device)

        tbar = tqdm.trange(self.n_epoch)
        for epoch in tbar:
            self.model.train()
            total_loss = .0
            for batch in self.loaders["train"]:
                self.optimizer.zero_grad()

                emb = self.encoder(batch.x)  # x is input atom feature (attributs de noeuds)
                # edge_emb = bond_encoder(batch.edge_attr)  # pour les arrêtes

                out = self.model(emb, batch.edge_index, batch.batch)

                loss = self.loss_func(out, batch.y.squeeze(1))
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            batch_mean_loss = total_loss / len(self.loaders["train"])
            self.eval(mode="valid")
            self.metrics["duration"].append(time.time() - time_start)
            self.metrics["loss"].append(batch_mean_loss)

            self.pocket(epoch)
            tbar.set_description(f'Current Valid Rocauc: {self.metrics["valid_rocauc"][-1]:.4f}, '
                                 f'Best Rocauc: {self.metrics["best_valid_rocauc"]}, '
                                 f'Batch Mean loss: {batch_mean_loss:.4f}')

        time_elapsed: .4
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
                y_true.append(batch.y.view(batch.y.shape))

                emb = self.encoder(batch.x)
                out = self.model(emb, batch.edge_index, batch.batch)[:, 1]
                y_pred.append(out.detach().cpu())

            y_true = torch.cat(y_true, dim=0).numpy()
            y_pred = torch.cat(y_pred, dim=0).unsqueeze(1).numpy()

            rocauc = self.evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['rocauc']
            if mode == "valid":
                # noinspection PyBroadException
                try:
                    self.metrics["valid_rocauc"].append(rocauc)
                except Exception as e:
                    self.metrics["valid_rocauc"] = [rocauc]
            else:
                self.metrics[f"{mode}_rocauc"] = rocauc

    def pocket(self, epoch):
        is_best_defined = "best_valid_rocauc" in self.metrics.keys()
        if (not is_best_defined) or self.metrics["valid_rocauc"][-1] > self.metrics["best_valid_rocauc"]:
            self.metrics["best_valid_rocauc"] = self.metrics["valid_rocauc"][-1]
            self.metrics["best_valid_epoch"] = epoch
            self.model.save("best_valid")


def main():
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

    parameters = {"emb_dim": 100,
                  "n_classes": int(dataset.num_classes),
                  "hidden_dim": 128,
                  "dropout": .4,
                  "batch_size": 64,
                  "learning_rate": 0.001,
                  "n_epoch": 100}

    model = MolhivModels.GCN3BN3MLP2(parameters=parameters)

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


def launch_expermients():
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

    '''
    models = [MolhivModels.SAGE3BN3MLP2,
              MolhivModels.SAGE3BN1MLP2,
              MolhivModels.GCN3BN3MLP2,
              MolhivModels.GCN3BN1MLP2,
              MolhivModels.GCN2MLP2]
    '''

    models = [MolhivModels.GCN3BN3MLP2,
              MolhivModels.GCN3BN1MLP2,
              MolhivModels.GCN2MLP2]


    n_epoch = 30
    parameters_set = {"best": {"emb_dim": 100,
                           "n_classes": int(dataset.num_classes),
                           "hidden_dim": 128,
                           "dropout": .4,
                           "batch_size": 64,
                           "learning_rate": 0.001,
                           "n_epoch": n_epoch},
                  "hidden_dim": {"emb_dim": 100,
                                 "n_classes": int(dataset.num_classes),
                                 "hidden_dim": 64,
                                 "dropout": .4,
                                 "batch_size": 64,
                                 "learning_rate": 0.001,
                                 "n_epoch": n_epoch},
                  "lr": {"emb_dim": 100,
                         "n_classes": int(dataset.num_classes),
                         "hidden_dim": 128,
                         "dropout": .4,
                         "batch_size": 64,
                         "learning_rate": 0.01,
                         "n_epoch": n_epoch}
                  }

    experiments = {}
    for model in models:
        for p_key in parameters_set.keys():
            experiments[model.__name__ + "_" + p_key] = {"model": model,
                                                         "parameters": parameters_set[p_key]}

    EXP = {}
    for e_key, experiment in experiments.items():
        print("Exp", e_key)

        print("experiment", experiment)

        model_class = experiment["model"]
        experiment["model"] = str(experiment["model"].__name__)
        experiment["results"] = {}

        nb_test = 1
        rocauc_mean = .0
        nb_success = 0
        for i in range(nb_test):
            # noinspection PyBroadException
            try:
                print(f"_{i}_")
                model = model_class(parameters=experiment["parameters"])

                trainer = Trainer(model, dataset, parameters=experiment["parameters"])
                metrics = trainer.train()
                experiment["results"][i] = {"status": "Success",
                                            "metrics": metrics}
                rocauc_mean += metrics["best_valid_rocauc"]
                nb_success += 1

                print(metrics)
                print()

            except Exception as e:
                experiment["results"][i] = {"status": "Error",
                                            "error": str(e)}

            if i == nb_test - 1 and nb_success > 0:
                experiment["results"]["rocauc_mean"] = rocauc_mean / nb_success
            EXP[e_key] = experiment
            import pprint
            pprint.pprint(EXP)
            with open("arxiv_res.txt", "w") as f:

                f.write(json.dumps(EXP, indent=4))

        print("_" * 40)
        print()



if __name__ == "__main__":
    #main()
    launch_expermients()
