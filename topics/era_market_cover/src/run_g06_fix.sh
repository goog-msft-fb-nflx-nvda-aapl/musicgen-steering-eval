#!/bin/bash
# re-run of the Qwen3-8B draw r2 validation stage after a full disk interrupted it, then VINet for the three replicate sets (single writer of vinet_identity.json)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; T=qwen3_8b_r2
./venv/bin/python src/70_prompt_set_gen.py --split validation --select results/g06_${T}_selected.json --seeds 0 1 2 --outdir gen_g06_$T > logs/g06_${T}_val_gen2.log 2>&1
GENDIR=gen_g06_$T ./venv/bin/python src/10_eval_gen.py --out results/g06_${T}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/g06_${T}_eval.log 2>&1
SEEDS=0,1,2 GENDIR=gen_g06_$T ./venv/bin/python src/13_content_metrics.py --out results/g06_${T}_content.json > logs/g06_${T}_content.log 2>&1
./venv/bin/python src/54_vinet_identity.py gen_g06_qwen3_8b_r2 gen_g06_qwen25_7b_r2 gen_g06_qwen3_8b_r3 > logs/g06_rep_vinet2.log 2>&1
touch g06_fix.done
