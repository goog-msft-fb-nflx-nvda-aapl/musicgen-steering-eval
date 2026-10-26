#!/bin/bash
# server: market-targeted covers on dataset B. usage: bash run_market.sh <gpu_a> <gpu_b>
cd ~/era_market; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY=./venv/bin/python
CUDA_VISIBLE_DEVICES=$1 $PY src/12_codec_resynth.py --ds B > logs/codec_B.log 2>&1
export CUDA_VISIBLE_DEVICES=$1
$PY src/02_embed.py --ds B --conds acccodec_norm > logs/emb_codec_mert_B.log 2>&1
$PY src/02b_embed_muq.py --ds B --conds acccodec_norm > logs/emb_codec_muq_B.log 2>&1
$PY src/04_probe.py --ds B --conds acc_norm acccodec_norm --out results/probe_codec_mert_B.json > logs/probe_codec_mert_B.log 2>&1
$PY src/04_probe.py --ds B --conds acc_norm acccodec_norm --prefix muq_ --out results/probe_codec_muq_B.json > logs/probe_codec_muq_B.log 2>&1
export GENDIR=gen_market
CUDA_VISIBLE_DEVICES=$1 $PY src/07_generate.py --ds B --n 24 --seeds 0 1 2 > logs/gen_market_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 $PY src/07_generate.py --ds B --n 24 --seeds 0 1 2 --reverse > logs/gen_market_b.log 2>&1
wait
CUDA_VISIBLE_DEVICES=$1 $PY src/37_eval_market.py --out results/market_eval.json > results/market_summary.txt 2> logs/market_eval.err
DS=B ERAS=neutral,US,UK,Brazil,Spain,Germany,Italy SEEDS=0,1,2 $PY src/13_content_metrics.py --out results/market_content.json > logs/market_content.log 2>&1
touch market.done
