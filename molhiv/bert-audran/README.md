# README TP - DETAILS DES RESULTATS
BERT Audran

## Comment le lancer

python3 train.py

## Dataset ogbn-arxiv

Pour ce dataset j'utilise l'evaluateur fournit et j'utilise l'accuracy.

Ci-dessous un tableau regroupant les différents résultats que j'ai obtenu pour cette tache. 
Tous les entrainements avaient : 100 epochs, l'optimiseur Adam, un learning rate de 0.01, la nll_loss, log softmax en sortie du modèle. Si pas spécifié : 3 couches de convolutions, 1 couches linéaires avec 25% de dropout et Relu, hidden dim des différentes layers à 128.
Les résultats sont la moyenne sur 3 entrainements du modèle.
La colonne sur les temps d'entrainements permet de comparer deux modèles et de voir si par exemple pour gagner 2% il faut augmenter le temps de 25%.
La dernière colonne permet de savoir en moyenne combien d'epoch il faut faire pour avoir les meilleurs résultats possible.

| Modèle                     | Moyenne de la précision sur la validation | Moyenne de la précision sur le test | Moyenne des temps d'entrainements | Meilleure epoch en moyenne |
|:---------------------------|:-----------------------------------------:|:-----------------------------------:|:---------------------------------:|:--------------------------:|
| New GCNConv avec 3 layers  |                   74.45                   |                70.55                |          63 (100 epochs)          |             94             |
| New SAGEConv avec 3 layers |                   70.7                    |                70.12                |          60 (100 epochs)          |             96             |
| New SAGEConv avec 2 layers |                   66.76                   |                65.74                |          57 (100 epochs)          |             94             |
| SAGEConv                   |                  62.41 %                  |               55.91 %               |         167 (250 epochs)          |            115             |
| GCNConv                    |                   61.96                   |                55.62                |         170 (250 epochs)          |            208             |

Descriptif plus détaillé des modèles:
- "New": correspond à l'utilisation de "data.adj_t = data.adj_t.to_symmetric()" dans le modèle, sans new cela veut dire utilisation des edges index. Le modèle a aussi un peu changé: n_layers*(conv->norm->relu) -> linear
- Les modèles anciens ont la même architecture que pour l'autre dataset
- La différence entre le SageConv et le GCN conv pourrait être dû à de la chance.

## Dataset ogbg-molhiv

Pour ce dataset il est nécessaire d'utiliser la métrique "rocauc" pour la précision.

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