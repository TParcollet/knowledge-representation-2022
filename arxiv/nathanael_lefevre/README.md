__Expériences sur Arxiv__

REMARQUE : Dû à des problèmes de disponibilité de GPU, Je ne présente les résultats que pour un seul test.

_Expérience 1:_ 
Pour cette première expérience, le modèle GCN2BN1LIN2MLP3 a été utilisé.
Une batch norm est appliquée au début du modèle.
Il est constitué de 2 couches GCNConv, relu et dropout après chaque convolution.
Suite aux convolutions, un MLP à 2 couches est appliqué.

*Résultats par paramètres:*
- parameters = {"features_dim": dataset.num_node_features,
                  "n_classes": int(dataset.num_classes),
                  "hidden_dim": 256,
                  "dropout": .2,
                  "batch_size": 256,
                  "learning_rate": 0.001,
                  "n_epoch": 20}

Meilleure epoch : 17
Meilleure accuracy de validation : 0.598117070985288
Accuracy de test : 0.5983582988478949


_Expérience 2:_
Pour cette première expérience, le modèle GCN2LIN2MLP3 a été utilisé.
Aucune batch norm n'est appliquée au début du modèle.
Il est constitué de 2 couches GCNConv, relu et dropout après chaque convolution.
Suite aux convolutions, un MLP à 2 couches est appliqué.

- parameters = {"features_dim": dataset.num_node_features,
                  "n_classes": int(dataset.num_classes),
                  "hidden_dim": 256,
                  "dropout": .2,
                  "batch_size": 256,
                  "learning_rate": 0.001,
                  "n_epoch": 20}

Meilleure epoch : 17
Meilleure accuracy de validation : 0.5924836002355383
Accuracy de test : 0.5924268034427154



*Conclusion* : La Batch Norm n'améliore pas significativement les résultats ici
