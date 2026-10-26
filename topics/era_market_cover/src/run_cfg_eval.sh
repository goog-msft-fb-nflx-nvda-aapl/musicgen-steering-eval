#!/bin/bash
# server: score guidance-scale sweep (seed 0, 24 songs): codec-matched era probes + content metrics; gen = scale 3 reference
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for g in 1 5 7 3; do
  d=gen_cfg$g; [ $g = 3 ] && d=gen
  GENDIR=$d ./venv/bin/python src/10_eval_gen.py --out results/cfg${g}_eval.json --train_cond acccodec_norm --seeds 0 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/cfg${g}_eval.log 2>&1
  SEEDS=0 GENDIR=$d ./venv/bin/python src/13_content_metrics.py --out results/cfg${g}_content.json > logs/cfg${g}_content.log 2>&1
done
touch cfg_eval.done
