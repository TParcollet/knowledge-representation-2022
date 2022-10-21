# README TP - DETAILS DES RESULTATS
BERT Audran

## Comment le lancer

```
python3 train.py
```

Dans le fichier train.py, juste après les import il y a le choix du nombre d'epochs et du nombre de trains.
Les autres paramètres sont dans le code.
Le requirements.txt contient les différents package.

## Dataset ogbn-arxiv

Pour ce dataset j'utilise l'evaluateur fourni et j'utilise l'accuracy.

Ci-dessous un tableau regroupant les différents résultats que j'ai obtenu pour cette tache. 
Tous les entrainements avaient : 100 epochs, l'optimiseur Adam, un learning rate de 0.01, la nll_loss, log softmax en sortie du modèle. Si pas spécifié : 3 couches de convolutions, 1 couches linéaires avec 25% de dropout et Relu, hidden dim des différentes layers à 128.
Les résultats sont la moyenne sur 3 entrainements du modèle (pas plus par manque de temps).
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

