#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/54_vinet_identity.py gen_lora_pd3s2 > logs/vinet4a.log 2>&1
until [ -f neg_g5.done ] && [ -f lora_lr3e4s2.done ]; do sleep 30; done
./venv/bin/python src/54_vinet_identity.py gen_neg_g5 gen_lora_lr3e4s2 > logs/vinet4b.log 2>&1
touch vinet4.done
