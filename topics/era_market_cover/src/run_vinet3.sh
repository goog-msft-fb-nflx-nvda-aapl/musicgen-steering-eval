#!/bin/bash
# VINet identity for generation dirs not yet scored; single process writes vinet_identity.json
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/54_vinet_identity.py gen_lora_nomels1 gen_lora_nomels2 gen_lora_pd3s1 gen_sp_t12k250 gen_sp_t14k250 > logs/vinet3a.log 2>&1
until [ -f neg_k250.done ]; do sleep 30; done
./venv/bin/python src/54_vinet_identity.py gen_neg_g3k250 > logs/vinet3b.log 2>&1
touch vinet3.done
