#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=msrs-ret-2round
#SBATCH --gres=gpu:1
#SBATCH --output=out_files/retrieve.out

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
  --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
source /ext3/env.sh;
conda activate base;

python3 retrieve.py \
  --domain story \
  --queries_path ../../data/story/queries_test.json \
  --documents_path ../../data/story/documents \
  --embeddings_path embeddings/story/contriever \
  --method llm_embedding \
  --model contriever \
  --two_round \
  --r1_model contriever \
  --r1_embeddings_path embeddings/story/contriever \
  --k_list 0 1 2 3 4 5 6 7 8 \
  --snippet_tokens 256;

python3 retrieve.py \
  --domain meeting \
  --queries_path ../../data/meeting/queries_test.json \
  --documents_path ../../data/meeting/documents \
  --embeddings_path embeddings/meeting/contriever \
  --method llm_embedding \
  --model contriever \
  --two_round \
  --r1_model contriever \
  --r1_embeddings_path embeddings/meeting/contriever \
  --k_list 1 2 3 \
  --snippet_tokens 256

python3 retrieve.py \
  --domain story \
  --queries_path ../../data/story/queries_test.json \
  --documents_path ../../data/story/documents \
  --embeddings_path embeddings/story/tuned-contriever \
  --method llm_embedding \
  --model tuned-contriever \
  --two_round \
  --r1_model contriever \
  --r1_embeddings_path embeddings/story/contriever \
  --k_list 0 1 2 3 4 5 6 7 8 \
  --snippet_tokens 256;

python3 retrieve.py \
  --domain meeting \
  --queries_path ../../data/meeting/queries_test.json \
  --documents_path ../../data/meeting/documents \
  --embeddings_path embeddings/meeting/tuned-contriever \
  --method llm_embedding \
  --model tuned-contriever \
  --two_round \
  --r1_model contriever \
  --r1_embeddings_path embeddings/meeting/contriever \
  --k_list 1 2 3 \
  --snippet_tokens 256
"
