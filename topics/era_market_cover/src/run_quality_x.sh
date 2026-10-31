#!/bin/bash
# Audiobox Aesthetics for the extra-decade sets. usage: bash run_quality_x.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ERAS=neutral,1970s,1990s,2000s
for n in trix_base_m3e7 trix_lora_m3e7 trix_lora_m3e10 trix_lora_m3e15v1 trix_lora_m3e20v1; do
  GENDIR=gen_$n ./venv2/bin/python src/18_quality.py --seeds 0 1 2 --out results/quality_$n.json > logs/quality_$n.log 2>&1
done
touch quality_x.done
