#!/bin/bash
cd ~/era_market; export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
(cd ext/ACE-Step-1.5; CUDA_VISIBLE_DEVICES=2 ~/era_market/venv_ace/bin/python ../../src/55_ace_cover.py --strength 1.0 --outdir ace_gen_c10 --seeds 0 1 2 > ../../logs/ace_gen_c10.log 2>&1)
CUDA_VISIBLE_DEVICES=2 GENDIR=ace_gen_c10 ./venv/bin/python src/10_eval_gen.py --out results/ace_c10_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/ace_c10_eval.log 2>&1
SEEDS=0,1,2 GENDIR=ace_gen_c10 ./venv/bin/python src/13_content_metrics.py --out results/ace_c10_content.json > logs/ace_c10_content.log 2>&1
CUDA_VISIBLE_DEVICES=2 ./venv/bin/python src/54_vinet_identity.py ace_gen_c02 ace_gen_c05 ace_gen_c08 ace_gen_c10 > logs/vinet_ace.log 2>&1
touch ace10.done
