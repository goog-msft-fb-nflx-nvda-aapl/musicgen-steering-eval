#!/bin/bash
# server launcher: bash run_stems.sh <gpu>  (separation then MERT-v2 embeddings for A and B)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for ds in A B; do ./venv/bin/python src/01_separate.py --ds $ds > logs/sep_$ds.log 2>&1; done
for ds in A B; do ./venv/bin/python src/02_embed.py --ds $ds > logs/emb_$ds.log 2>&1; done
touch stems_emb.done
