#!/bin/bash
# CLAP zero-shot era evaluator (independent of the codec-matched probes) for newer sets. usage: run_clap3.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for n in neg_g3 neg_g5 neg_g5k250 delta_g3 delta_g5 delta_g3k250 lora_nomel lora_nomels1 lora_nomels2 lora_lr3e4s2 lora_pd3s2 lora_v1s1 g06_qwen3_8b g06_qwen25_7b g06_qwen25_14b g06_llama32_1b; do
  GENDIR=gen_$n ./venv_mcl/bin/python src/26_clap_era_env.py --out results/clap_era_$n.json > logs/clap_$n.log 2>&1
done
for n in sp_t12k250 sp_t14k250 sp_k250; do GENDIR=gen_$n ./venv_mcl/bin/python src/26_clap_era_env.py --seeds 0 1 --out results/clap_era_$n.json > logs/clap_$n.log 2>&1; done
touch clap3.done
