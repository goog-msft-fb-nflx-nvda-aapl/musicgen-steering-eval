#!/bin/bash
# usage: run_lora_seed.sh <gpu> <seed>  : v1 recipe (r16, 3000 steps, bs 8, lr 1e-4) with another training seed, then generation and scoring
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name v1s$2 --r 16 --steps 3000 --bs 8 --lr 1e-4 --seed $2 > logs/lora_v1s$2.log 2>&1
GPU=$1 V=v1s$2 G=gen_lora_v1s$2 P=lora_v1s$2 ./run_lora_gen.sh
