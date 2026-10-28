#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for g in 3 5; do
  ./venv/bin/python src/60_delta_guidance.py --g $g --outdir gen_delta_g$g --seeds 0 1 2 > logs/delta_g$g.log 2>&1
  GENDIR=gen_delta_g$g ./venv/bin/python src/10_eval_gen.py --out results/delta_g${g}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/delta_g${g}_eval.log 2>&1
  SEEDS=0,1,2 GENDIR=gen_delta_g$g ./venv/bin/python src/13_content_metrics.py --out results/delta_g${g}_content.json > logs/delta_g${g}_content.log 2>&1
  ./venv/bin/python src/54_vinet_identity.py gen_delta_g$g > logs/vinet_delta$g.log 2>&1
done
touch delta.done
