#!/bin/bash
# LoRA (v1 adapters) combined with contrast guidance. usage: run_lora_guid.sh <gpu> <delta|neg> <g>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 LORA=v1 NEUTRAL_SRC=gen_lora
k=$2; g=$3; n=gen_lorav1_${k}_g$g; sc=src/60_delta_guidance.py; [ $k = neg ] && sc=src/60b_negprompt_guidance.py
./venv/bin/python $sc --g $g --outdir $n --seeds 0 1 2 > logs/lorav1_${k}_g$g.log 2>&1
GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/lorav1_${k}_g${g}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/lorav1_${k}_g${g}_eval.log 2>&1
SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/lorav1_${k}_g${g}_content.json > logs/lorav1_${k}_g${g}_content.log 2>&1
touch lorav1_${k}_g$g.done
