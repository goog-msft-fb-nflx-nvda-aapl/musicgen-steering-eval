#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name v3 --r 4 --steps 3000 --bs 8 --lr 1e-4 > logs/lora_v3.log 2>&1
GPU=1 V=v3 G=gen_lora_v3 P=lora_v3 ./run_lora_gen.sh
