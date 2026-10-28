#!/bin/bash
# server: sampling-parameter ablation (G07), seeds 0 1, then scoring. usage: run_sp.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for c in k250 t08 t12 k0p09; do
  ./venv/bin/python src/68_sampling_params.py --config $c --outdir gen_sp_$c --seeds 0 1 > logs/sp_$c.log 2>&1
  GENDIR=gen_sp_$c ./venv/bin/python src/10_eval_gen.py --out results/sp_${c}_eval.json --train_cond acccodec_norm --seeds 0 1 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/sp_${c}_eval.log 2>&1
  SEEDS=0,1 GENDIR=gen_sp_$c ./venv/bin/python src/13_content_metrics.py --out results/sp_${c}_content.json > logs/sp_${c}_content.log 2>&1
done
touch sp.done
