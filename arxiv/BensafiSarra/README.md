Enseignant Titouan Parcollet
Etudiante : Bensafi Sarra
Master 2  IA-Ilsen 
Année 2022-2023

Dataset Arxiv :

Code source : https://colab.research.google.com/drive/1i-efgCUincDdVXG06YIeQzEkgFLRpPBN?usp=sharing
Nous utiliserons le réseau ogbn-arxiv dans lequel chaque nœud est un article d'informatique sur l'arxiv et chaque bord dirigé représente qu'un article en cite un autre. La tâche consiste à classer chaque nœud dans une classe d'article (40 classes).

Model                    | Résultat
GCNConv x 4              | 0.688
GCNConv x 2              | 0.687
GCNConv + Sage + GCNConv | 0.663


Dans le modèle ou j’ai eu une accuracy de 0.688,  j’ai empilé 4 couches GCNConv le nombre d’unité caché est de 256 (car cela a donné de meilleur résultats).
Entre chaque couche de GCNConv j’ai ajouté une BatchNorm, une fonction Relu puis un dropout a la fin du modèle j’ai utilisé une softmax puisque nous avons un problème de classifications multi-class ( 40 classes).
En ce qui concerne la fonction de coût j’ai utilisé la  nll_loss car elle n’applique le log-SoftMax à l'intérieur contrairement à l’entropie.
