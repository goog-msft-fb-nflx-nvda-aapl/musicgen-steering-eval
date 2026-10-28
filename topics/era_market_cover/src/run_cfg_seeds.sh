#!/bin/bash
# server: CFG sweep (guidance 1, 5, 7; 3 already has 3 seeds in gen/) with seeds 1 and 2, then scoring with seeds 0 1 2. usage: run_cfg_seeds.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for g in 1 5 7; do
  GENDIR=gen_cfg$g ./venv/bin/python src/07_generate.py --n 24 --seeds 1 2 --guidance $g > logs/gen_cfg${g}_s12.log 2>&1
  GENDIR=gen_cfg$g ./venv/bin/python src/10_eval_gen.py --out results/cfg${g}_s3_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/cfg${g}_s3_eval.log 2>&1
  SEEDS=0,1,2 GENDIR=gen_cfg$g ./venv/bin/python src/13_content_metrics.py --out results/cfg${g}_s3_content.json > logs/cfg${g}_s3_content.log 2>&1
done
touch cfg_seeds.done
