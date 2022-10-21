# README TP - DETAILS DES RESULTATS
BERT Audran

## Comment le lancer

```
python3 train.py
```

Dans le fichier train.py, juste après les import il y a le choix du nombre d'epochs et du nombre de trains.
Les autres paramètres sont dans le code.
Le requirements.txt contient les différents package.

## Dataset ogbg-molhiv

Le but de ce dataset est de prédire les propriétés des molécules (2 classes): soit la molécule inhibe la réplication  du virus du VIH ou non.

Pour ce dataset il est nécessaire d'utiliser la métrique "rocauc" pour la précision, j'utilise donc l'evaluateur foruni.

Ci-dessous un tableau regroupant les différents résultats que j'ai obtenu pour cette tache. 
Tous les entrainements avaient : 100 epochs, l'optimiseur Adam, un learning rate de 0.01, la nll_loss, log softmax en sortie du modèle. Si pas spécifié : 2 couches de convolutions, 2 couches linéaires avec 25% de dropout et Relu, hidden dim des différentes layers à 128.
Les résultats sont la moyenne sur 5 entrainements du modèle.
La colonne sur les temps d'entrainements permet de comparer deux modèles et de voir si par exemple pour gagner 2% il faut augmenter le temps de 25%.
La dernière colonne permet de savoir en moyenne combien d'epoch il faut faire pour avoir les meilleurs résultats possible.

| Modèle            | Moyenne de la précision sur la validation | Moyenne de la précision sur le test | Moyenne des temps d'entrainements |                            Meilleure epoch en moyenne                             |
|:------------------|:-----------------------------------------:|:-----------------------------------:|:---------------------------------:|:---------------------------------------------------------------------------------:|
| SAGEConv + enc    |                  82.21 %                  |               79.08 %               |              400sec               |                                        82                                         |
| GCNConv + enc     |                   80.59                   |                76.57                |                424                |          77 (ici la variance est grande il y a eu 44 et 100 par exemple)          |
| GINConv + enc     |                   79.35                   |                74.84                |                415                |                               69 (grande variance)                                |
| GATv2Conv + enc   |                   74.45                   |                 62                  |                595                |                                                                                   |
| GCNConv           |                   70.74                   |                64.59                |                347                |                               65 (grande variance)                                |

Descriptif plus détaillé des modèles:
- SageConv: Ce modèle utilise les couches SageConv pour la convolution. Encoder de noeud (embedding de taille=100) -> SageConv -> batch norm -> MLP à 2 couches (les autres modèles ont la même forme)
- GATv2Conv: je sais pas pourquoi les résultats sont si bas, je dois surement mal faire les choses
- GCNConv: on voit que l'encodeur de noeuds améliore grandement les perfs

## Sources

https://ogb.stanford.edu/docs/graphprop/#ogbg-mol
https://pytorch-geometric.readthedocs.io/en/latest/
https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py
