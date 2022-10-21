Enseignant Titouan Parcollet
Etudiante : Bensafi Sarra
Master 2  IA-Ilsen 
Année 2022-2023

Dataset Molhiv :

Code Source : https://colab.research.google.com/drive/13jW3XaN_hf1PUw9x6IpXCgkmpLvIqJo1?usp=sharing

Nous utiliserons des données ou le graphe qui représente des molécules, Il est adopté à partir de MoleculeNet [1] et fait partie des ensembles de données MoleculeNet les plus importants.  On veut savoir si une molécule inhibe ou non la réplication du virus VIH c'est-à-dire on va faire une classification binaire.

Model                       | Résultats (Accuracy)
GCNConv +  2 linear layer   | 0.651
GCNConv + Sage + 2 linear   | 0.737
GATConv + 2 linear          | 0.667
SageConv + 2 linear         | 0.710
GCNConv + TAGConv + 2 linear| 0.747


Dans le meilleur modèle j’ai utilisé une couche GCN et une couche TAGConv en ajoutant deux couches linéaires à la fin, à la sortie du modèle je n’ai pas utilisé une sigmoid puisque j’ai utilisé la fonction de coût BCEWithLogitsLoss qui inclut à l'intérieur une sigmoid pour la classification binaire. Après les couches GCN et TAG j’ai ajouté une couche BatchNorm puis une Relu avec un dropout de 0.5
