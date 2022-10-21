```bash
rm resultogbg-molhiv-final.txt
sbatch ./runtrain.sh ogbg-molhiv-final --dataset=ogbg-molhiv --epochs=80 --task="graph" --gnn-depth=3 --emb-size=64 --atom-emb-size=64`
# wait for task to finish
./avgscore.sh resultogbg-molhiv-final.txt
```

The final model uses 4 conv layers with a node embedding size of 64. Training was made on one 11GB 2080Ti for about 2.5 minutes over 80 epochs. Automatic mixed precision (AMP) was used in order to improve training time. Training was monitored using TensorBoard.

The applied convolution is custom and is based on PyTorch Geometric's `MessagePassing` facility. It makes use of node (atom) features as well as the edge (bond) features, with embeddings generated with OGB's provided utils. Self-loops are added.

Over 10 model trains, the ROCAUC score on the test dataset was 73.9% on average, with the worst result being 71.7% and the best result being 78.7%.  
The observed variance over different trains was much greater than the OGB leaderboard suggested. Training over the `ogbn-arxiv` dataset, on the other hand, generally led to the similar performance over many runs. This made it generally difficult to find improvements to the model. Throughout experimentation, I could not find improvements that improved my results significantly.

## Observations

### BatchNorm

BatchNorm (here applied after aggregation in each convolution) made the model converge much faster consistently.

### Aggregation method

Multiple aggregation methods were attempted. The final model makes use of multiple aggregation methods concatenated, as this improved the model performance somewhat, at the cost of model size and training time: mean aggregation, standard aggregation, and an attention-based aggregation.  
This results in a vector that is 3 times the embedding size, which is then mapped back to an embedding vector of size 64 after a simple non-linear transform.

Attention-based aggregation computes the aggregation score using a simple linear map, and node features are transformed with one simple linear layer. Adding complexity to the attention-based aggregation did not improve results, and made the model slower to train.

### Activation functions

Throughout the model, the hardswish activation function was used. Using ReLU instead did not have a significant impact on model performance.

### Embedding size

In most model topologies tried, an embedding size higher than 64 increased model complexity and slowed down training with no significant improvement to model performance.

### Graph convolution depth

When increased beyond 3, graph convolution depth appeared to worsen model performance, or at least slowed down model training with no benefit.

### Batch size

A batch size of 2048 sped up training significantly. Using more conservative batch sizes (e.g. of size 64) did not improve results, and made training much slower, and in fact made training less stable (as in the measured accuracy was swaying). It could be that batch normalization benefited from higher batch sizes, and that higher batch sizes are made possible by the batch normalization.

### Optimizer, LR and LR schedule

The AdamW optimizer was used, with a LR of 0.001. For the most part, changing the base LR did not have a significant impact on training, but 0.001 appeared to be a good balance between convergence speed and training stability.

Using different LR schedulers (cosine annealing, linear decay, LR reducing on loss plateau) did not have the intuitively expected "fine-tuning" effect and generally slowed down training or did not improve results. The final model makes use of no LR scheduler.
