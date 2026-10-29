#!/bin/bash
# negative-prompt guidance g=3 with top_k 250 (VINet identity is run afterwards by run_vinet3.sh to avoid concurrent writes to vinet_identity.json)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
n=gen_neg_g3k250
./venv/bin/python src/60b_negprompt_guidance.py --g 3 --top_k 250 --outdir $n --seeds 0 1 2 > logs/neg_g3k250.log 2>&1
GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/neg_g3k250_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/neg_g3k250_eval.log 2>&1
SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/neg_g3k250_content.json > logs/neg_g3k250_content.log 2>&1
touch neg_k250.done
