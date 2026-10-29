#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
until [ -f g06_qwen3_8b_r2.done ] && [ -f g06_qwen25_7b_r2.done ] && [ -f g06_qwen3_8b_r3.done ]; do sleep 60; done
./venv/bin/python src/54_vinet_identity.py gen_g06_qwen3_8b_r2 gen_g06_qwen25_7b_r2 gen_g06_qwen3_8b_r3 > logs/g06_rep_vinet.log 2>&1
touch g06_rep_vinet.done
