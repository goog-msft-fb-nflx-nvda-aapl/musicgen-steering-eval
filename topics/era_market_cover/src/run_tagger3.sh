#!/bin/bash
# Discogs-EffNet evaluator (CPU, venv_ess) for key newer generation sets, one process at a time (server CPU is shared)
cd ~/era_market; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 CUDA_VISIBLE_DEVICES=
for n in neg_g5 lora_nomels2 g06_qwen3_8b delta_g5 lora_pd3s2 lora_lr3e4s2; do
  ./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_$n --out results/discogs_$n.json > logs/discogs_$n.log 2>&1
done
touch tagger3.done
