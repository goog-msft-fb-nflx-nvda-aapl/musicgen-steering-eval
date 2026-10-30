#!/bin/bash
# Discogs-EffNet evaluator (CPU, venv_ess) for three-branch and LoRA+guidance sets, one process at a time (server CPU is shared)
cd ~/era_market; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 CUDA_VISIBLE_DEVICES=
for n in tri_lora_m3e7 tri_lora_m3e10 tri_base_m3e5 lorav1_neg_g3 lora_v1s1; do
  ./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_$n --out results/discogs_$n.json > logs/discogs_$n.log 2>&1
done
touch tagger4.done
