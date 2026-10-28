#!/bin/bash
# server: prompt elaboration ablation (word, short, tags) then scoring. usage: run_pv.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for v in word short tags; do
  ./venv/bin/python src/65_prompt_variants.py --variant $v --outdir gen_pv_$v --seeds 0 1 2 > logs/pv_$v.log 2>&1
  GENDIR=gen_pv_$v ./venv/bin/python src/10_eval_gen.py --out results/pv_${v}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/pv_${v}_eval.log 2>&1
  SEEDS=0,1,2 GENDIR=gen_pv_$v ./venv/bin/python src/13_content_metrics.py --out results/pv_${v}_content.json > logs/pv_${v}_content.log 2>&1
done
touch pv.done
