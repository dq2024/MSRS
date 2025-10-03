#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=msrs-ret
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/retrieve.out

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9; 

domain=story
python3 retrieve.py \
    --domain $domain \
    --queries_path ../../data/$domain/queries_test.json \
    --documents_path ../../data/$domain/documents \
    --embeddings_path embeddings/$domain/bm25 \
    --method bm25 \
    --model bm25
"