#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/61_lalm_era.py --out results/lalm_era_b3.json --gens gen_neg_g3 gen_sp_t12k250 gen_sp_t14k250 gen_lora_nomels1 gen_lora_pd3s1 > logs/lalm_b3.log 2>&1
touch judge3.done
