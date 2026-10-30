#!/bin/bash
# three-branch guidance (standard melody CFG + era contrast). usage: run_tri_ad.sh <gpu> <base|lora> <gm> <ge> [top_k] [adapter dir name under lora/, default v1]
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
m=$2; gm=$3; ge=$4; tk=${5:-50}; ad=${6:-v1}; n=gen_tri_${m}_m${gm}e${ge}${ad}
if [ $m = lora ]; then export LORA=$ad NEUTRAL_SRC=gen_lora; [ $ad = v1 ] || NEUTRAL_SRC=gen_lora_$ad; fi
./venv/bin/python src/61_triple_guidance.py --gm $gm --ge $ge --top_k $tk --outdir $n --seeds 0 1 2 > logs/tri_${m}_m${gm}e${ge}${ad}.log 2>&1
GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/tri_${m}_m${gm}e${ge}${ad}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/tri_${m}_m${gm}e${ge}${ad}_eval.log 2>&1
SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/tri_${m}_m${gm}e${ge}${ad}_content.json > logs/tri_${m}_m${gm}e${ge}${ad}_content.log 2>&1
touch tri_${m}_m${gm}e${ge}${ad}.done
