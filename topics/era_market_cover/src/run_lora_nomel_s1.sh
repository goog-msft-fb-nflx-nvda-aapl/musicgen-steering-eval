#!/bin/bash
# training-seed replicate of the no-chroma LoRA and of prompt dropout 0.3 (seed 1). usage: run_lora_nomel_s1.sh <gpu>
cd ~/era_market
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/48_lora_musicgen.py --name nomels1 --r 16 --steps 3000 --bs 8 --lr 1e-4 --no_melody --seed 1 > logs/lora_nomels1.log 2>&1
GENDIR=gen_lora_nomels1 LORA=nomels1 ./venv/bin/python src/07_generate.py --ds A --eras neutral 1960s 1980s 2010s --n 24 --seeds 0 1 2 --no_melody > logs/lora_nomels1_gen.log 2>&1
GENDIR=gen_lora_nomels1 ./venv/bin/python src/10_eval_gen.py --out results/lora_nomels1_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/lora_nomels1_eval.log 2>&1
SEEDS=0,1,2 GENDIR=gen_lora_nomels1 ./venv/bin/python src/13_content_metrics.py --out results/lora_nomels1_content.json > logs/lora_nomels1_content.log 2>&1
./run_lora_abl.sh $1 pd3s1 --pdrop 0.3 --seed 1
touch lora_nomels1.done
