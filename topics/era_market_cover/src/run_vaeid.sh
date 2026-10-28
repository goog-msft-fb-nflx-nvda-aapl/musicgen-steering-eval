#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
GENDIR=ace_vae_id ./venv/bin/python src/10_eval_gen.py --out results/ace_vae_id_eval.json --train_cond acccodec_norm --seeds 0 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/ace_vae_id_eval.log 2>&1
SEEDS=0 GENDIR=ace_vae_id ./venv/bin/python src/13_content_metrics.py --out results/ace_vae_id_content.json > logs/ace_vae_id_content.log 2>&1
./venv/bin/python src/54_vinet_identity.py ace_vae_id > logs/vinet_vae.log 2>&1
touch vaeid.done
