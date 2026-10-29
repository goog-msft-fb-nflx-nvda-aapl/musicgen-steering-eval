#!/bin/bash
# negative-prompt guidance g=5 (top_k 50); VINet identity afterwards in run_vinet4.sh
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
n=gen_neg_g5
./venv/bin/python src/60b_negprompt_guidance.py --g 5 --outdir $n --seeds 0 1 2 > logs/neg_g5.log 2>&1
GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/neg_g5_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/neg_g5_eval.log 2>&1
SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/neg_g5_content.json > logs/neg_g5_content.log 2>&1
touch neg_g5.done
