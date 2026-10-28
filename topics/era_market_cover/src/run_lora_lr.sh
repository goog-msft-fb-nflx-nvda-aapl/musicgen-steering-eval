#!/bin/bash
# G11c: learning-rate ablation at r16, 3000 steps (v1 = lr 1e-4). usage: run_lora_lr.sh <gpu> <name> <lr>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name $2 --r 16 --steps 3000 --bs 8 --lr $3 > logs/lora_$2.log 2>&1
GPU=$1 V=$2 G=gen_lora_$2 P=lora_$2 ./run_lora_gen.sh
touch lora_$2.done
