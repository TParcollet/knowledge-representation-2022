#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH --job-name="traingnn"
#SBATCH --gpus=1
#SBATCH --constraint=GPURAM_Max_12GB
#SBATCH --exclude=talos,aura

comment=$1

source ~/.bashrc
conda activate torchgeometric
for i in {1..10}; do
    srun python run.py --device=cuda "${comment}${i}" ${@:2} >> result${comment}.txt
done