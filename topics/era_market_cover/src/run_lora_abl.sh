#!/bin/bash
# G11c: usage run_lora_abl.sh <gpu> <name> <extra training args...>  (r16, 3000 steps, lr 1e-4 unless overridden)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; g=$1; n=$2; shift 2
./venv/bin/python src/48_lora_musicgen.py --name $n --r 16 --steps 3000 --bs 8 --lr 1e-4 "$@" > logs/lora_$n.log 2>&1
GPU=$g V=$n G=gen_lora_$n P=lora_$n ./run_lora_gen.sh
touch lora_$n.done
