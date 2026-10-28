#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/10_eval_gen.py --out results/cfg3_s3_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/cfg3_s3_eval.log 2>&1
SEEDS=0,1,2 ./venv/bin/python src/13_content_metrics.py --out results/cfg3_s3_content.json > logs/cfg3_s3_content.log 2>&1
touch cfg3_s3.done
