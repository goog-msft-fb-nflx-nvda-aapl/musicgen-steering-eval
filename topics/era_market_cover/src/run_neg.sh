#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for g in 3; do
  ./venv/bin/python src/60b_negprompt_guidance.py --g $g --outdir gen_neg_g$g --seeds 0 1 2 > logs/neg_g$g.log 2>&1
  GENDIR=gen_neg_g$g ./venv/bin/python src/10_eval_gen.py --out results/neg_g${g}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/neg_g${g}_eval.log 2>&1
  SEEDS=0,1,2 GENDIR=gen_neg_g$g ./venv/bin/python src/13_content_metrics.py --out results/neg_g${g}_content.json > logs/neg_g${g}_content.log 2>&1
  ./venv/bin/python src/54_vinet_identity.py gen_neg_g$g > logs/vinet_neg$g.log 2>&1
done
touch neg.done
