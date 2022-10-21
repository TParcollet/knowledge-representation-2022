```bash
rm resultogbn-arxiv-final.txt
sbatch ./runtrain.sh ogbn-arxiv-final --dataset=ogbn-arxiv --epochs=1000 --task="node" --gnn-depth=3 --emb-size=64
# wait for task to finish
./avgscore.sh resultogbn-arxiv-final.txt
```

The final model uses 4 conv layers with a node embedding size of 64. Training was made on one 11GB 2080Ti for about 6 minutes per 1000 epochs. Automatic mixed precision (AMP) was used in order to improve training time. Training was monitored using TensorBoard.

The applied convolution is custom and is based on PyTorch Geometric's `MessagePassing` facility. Self-loops are added.

Over 10 model trains, accuracy on the test dataset was 69.9% on average, with the worst result being 69.9% and the best result being 71.2%.  
Model performance was generally consistent across runs. Throughout experimentation, I could not find improvements that improved my results significantly.

## Observations

### BatchNorm

BatchNorm (here applied after aggregation in each convolution) made the model converge much faster consistently. On this dataset, accuracy began to plateau after about 100 epochs, whereas with different topologies it usually took several hundreds of epochs to reach similar accuracy.

### Aggregation method

Multiple aggregation was not found to be helpful here. Attentional aggregation is being used, with a rationale regarding the transforms that are applied being described in `README-molhiv.md`.

Attentional aggregation led to longer training time and memory usage, hence limiting the embedding size and convolution depth as a result. At the end, this was not a significant concern because increasing these two hyperparameters did not improve results.

### Activation functions

Throughout the model, the hardswish activation function was used. Using ReLU instead did not have a significant impact on model performance.

### Embedding size

In most model topologies tried, an embedding size higher than 64 increased model complexity and slowed down training with no significant improvement to model performance.

### Convolution depth

A graph convolution depth of 3 led to similar results as higher models.

### Batch size

No batch size is being used; the entire graph is being fed to the model during training or inference.

### Other convolutions

Stock PyTorch Geometric `GCNConv` and `GATv2Conv` were tried. The custom convolution seemed to slightly outperform these two in my particular configuration.

### Optimizer, LR and LR schedule

The AdamW optimizer was used, with a LR of 0.001. Adjusting the LR and LR scheduling was not spent a lot of time on for this dataset, due to unpromising results with the `molhiv` dataset.

### Augmentation

A dropout (p=0.1) is applied over the adjacency matrix, i.e. randomly dropping about 10% of graph connections randomly during every epoch. The intuition was that it would help the model generalize better. This appears to slightly improve accuracy. Higher values (e.g. p=0.25) only made it more difficult for the model to converge.
