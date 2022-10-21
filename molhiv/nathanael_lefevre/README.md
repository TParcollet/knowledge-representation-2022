__Expériences sur Molhiv__

Les résultats complets des expériences sont disponibles dans les fichier molhiv_res.txt 
et molhiv_res_GCN3BN1MLP2.txt

REMARQUE : Dû à des problèmes de disponibilité de GPU, je n'ai pas pu faire de moyenne sur chaque paramètre,
pour chaque expérience. Je propose donc un "potentiel moyen" pour chaque modèle qui fais la moyenne des résultats
pour chaque paramètres...

_Expérience 1:_ 
Pour cette première expérience, le modèle GCN3BN3MLP2 a été utilisé.
Il est constitué de 3 couches GCNConv avec batchNorm, relu et dropout après chaque convolution.
Suite aux convolutions, une global_mean_pool est appliquée puis un MLP à 2 couches est appliqué.

*Résultats par paramètres:*
parameters': {'batch_size': 64,
             'dropout': 0.4,
             'emb_dim': 100,
             'hidden_dim': 128,
             'learning_rate': 0.001,
             'n_classes': 2,
             'n_epoch': 30},

Meilleure epoch : 23
Meilleure rocauc de validation : 0.7720274838330392
Rocauc de test : 0.7071940361922786

- parameters: {'batch_size': 64,
               'dropout': 0.4,
               'emb_dim': 100,
               'hidden_dim': *64*,
               'learning_rate': 0.001,
               'n_classes': 2,
               'n_epoch': 30}

meilleure epoch : 24
Meilleure rocauc de validation : 0.7602635704487556
Rocauc de test : 0.7066996272620174

- parameters: {'batch_size': 64,
             'dropout': 0.4,
             'emb_dim': 100,
             'hidden_dim': 128,
             'learning_rate': *0.001*,
             'n_classes': 2,
             'n_epoch': 30}

meilleure epoch : 23
Meilleure rocauc de validation : 0.7720274838330392
Rocauc de test : 0.7071940361922786


*Conclusion* : Pour cette architecture, les résultats sont très stable et on peut se permettre de mêttre des couches 
de taille plus petite.
On a un potentiel moyen de rocauc de 0.7070292332155249
La batch Norm permet d'avoir de meilleurs résultats


_Expérience 2:_ 
Pour cette seconde expérience, le modèle GCN3BN1MLP2 a été utilisé.
Une seule batch Norm est appliquée au début du modèle.
Il est constitué de 3 couches GCNConv avec relu et dropout après chaque convolution.
Suite aux convolutions, une global_mean_pool est appliquée puis un MLP à 2 couches est appliqué.

*Résultats par paramètres:*

- parameters: {"emb_dim": 100,
                "n_classes": 2,
                "hidden_dim": 128,
                "dropout": 0.4,
                "batch_size": 64,
                "learning_rate": 0.001,
                "n_epoch": 30}

meilleure epoch : 14
Meilleure rocauc de validation : 0.7476025132275131
Rocauc de test : 0.7371772340137892


- parameters: {"emb_dim": 100,
            "n_classes": 2,
            "hidden_dim": 64,
            "dropout": 0.4,
            "batch_size": 64,
            "learning_rate": 0.001,
            "n_epoch": 30}

meilleure epoch : 26
Meilleure rocauc de validation : 0.7476025132275131
Rocauc de test : 0.7217404739373106


- parameters: {"emb_dim": 100,
            "n_classes": 2,
            "hidden_dim": 128,
            "dropout": 0.4,
            "batch_size": 64,
            "learning_rate": 0.01,
            "n_epoch": 30}

meilleure epoch : 26
Meilleure rocauc de validation : 0.6253475284146581
Rocauc de test : 0.5855404700747407


*Conclusion* : Pour cette seconde expérience, Les résultats sont moins stable.
Avec un learning rate trop élevé, on apprends très mal. Nous ne prendrons donc pas en compte le test avec lr=0.01
On a un potentiel moyen de rocauc de 0.7294588539755499



_Expérience 3:_ 
*Meileurs résultats*
Pour cette troisième expérience, le modèle GCN2MLP2 a été utilisé.
Aucune batch Norm n'est appliquée dans le modèle
Il est constitué de 2 couches GCNConv avec relu et dropout après chaque convolution.
Suite aux convolutions, une global_mean_pool est appliquée puis un MLP à 2 couches est appliqué.

*Résultats par paramètres:*

- parameters: {
    "emb_dim": 100,
    "n_classes": 2,
    "hidden_dim": 128,
    "dropout": 0.4,
    "batch_size": 64,
    "learning_rate": 0.001,
    "n_epoch": 30
}

meilleure epoch : 23
Meilleure rocauc de validation : 0.7161289927493631
Rocauc de test : 0.6693060893412388


- parameters: {
    "emb_dim": 100,
    "n_classes": 2,
    "hidden_dim": 64,
    "dropout": 0.4,
    "batch_size": 64,
    "learning_rate": 0.001,
    "n_epoch": 30
}

meilleure epoch : 22
Meilleure rocauc de validation : 0.7139014427787576
Rocauc de test : 0.6543733946194403


- parameters: {
    "emb_dim": 100,
    "n_classes": 2,
    "hidden_dim": 128,
    "dropout": 0.4,
    "batch_size": 64,
    "learning_rate": 0.01,
    "n_epoch": 30
}

meilleure epoch : 22
Meilleure rocauc de validation : 0.5926507691553988
Rocauc de test : 0.621132119198903


*Conclusion*: Pour cette architecture, les résultats sont moins stable et des couches de taille plus petite 
entraînent une perte de performance.
On a un potentiel moyen de rocauc de 0.6482705343865274


*Conclusion générale*
Une seule batch Norm suffit et aide à améliorer les résultats
L'utilisation de la global_mean_pool a également aidé à améliorer les résultats mais je n'ai plus de test sans.