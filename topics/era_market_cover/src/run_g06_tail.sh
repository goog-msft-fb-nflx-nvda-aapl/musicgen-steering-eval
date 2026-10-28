#!/bin/bash
# G06 from the selection step onward (candidate generation/eval already done). usage: run_g06_tail.sh <gpu> <tag>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; T=$2
./venv/bin/python src/71_select_prompts.py $T > logs/g06_${T}_select.log 2>&1
./venv/bin/python src/70_prompt_set_gen.py --split validation --select results/g06_${T}_selected.json --seeds 0 1 2 --outdir gen_g06_$T > logs/g06_${T}_val_gen.log 2>&1
GENDIR=gen_g06_$T ./venv/bin/python src/10_eval_gen.py --out results/g06_${T}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/g06_${T}_eval.log 2>&1
SEEDS=0,1,2 GENDIR=gen_g06_$T ./venv/bin/python src/13_content_metrics.py --out results/g06_${T}_content.json > logs/g06_${T}_content.log 2>&1
./venv/bin/python src/54_vinet_identity.py gen_g06_$T > logs/g06_${T}_vinet.log 2>&1
touch g06_$T.done
