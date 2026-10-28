#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name v2 --steps 1000 --bs 8 --lr 1e-4 > logs/lora_v2.log 2>&1
V=v2 G=gen_lora_v2 P=lora_v2 ./run_lora_gen.sh
