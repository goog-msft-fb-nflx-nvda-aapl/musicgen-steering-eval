#!/bin/bash
# no-chroma LoRA, training seed 2. usage: run_lora_nomel_s2.sh <gpu>
cd ~/era_market
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name nomels2 --r 16 --steps 3000 --bs 8 --lr 1e-4 --no_melody --seed 2 > logs/lora_nomels2.log 2>&1
GENDIR=gen_lora_nomels2 LORA=nomels2 ./venv/bin/python src/07_generate.py --ds A --eras neutral 1960s 1980s 2010s --n 24 --seeds 0 1 2 --no_melody > logs/lora_nomels2_gen.log 2>&1
GENDIR=gen_lora_nomels2 ./venv/bin/python src/10_eval_gen.py --out results/lora_nomels2_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/lora_nomels2_eval.log 2>&1
SEEDS=0,1,2 GENDIR=gen_lora_nomels2 ./venv/bin/python src/13_content_metrics.py --out results/lora_nomels2_content.json > logs/lora_nomels2_content.log 2>&1
touch lora_nomels2.done
