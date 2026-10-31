#!/bin/bash
# 2000s prompt-wording variants with the LoRA, three-branch guidance. usage: run_tri_p00.sh <gpu> <ge>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 LORA=v1 NEUTRAL_SRC=gen_lora EXTRA_PROMPTS=$HOME/era_market/src/prompts_2000s_variants.json
ge=$2; n=gen_trp00_lora_m3e$ge; tag=trp00_lora_m3e$ge
./venv/bin/python src/61_triple_guidance.py --gm 3 --ge $ge --eras 2000s_a 2000s_b 2000s_c 2000s_d --outdir $n --seeds 0 1 2 > logs/$tag.log 2>&1
for f in gen_lora/*_neutral_s[012]_acc.wav gen_lora/*_neutral_s[012]_remix.wav; do ln -sf ~/era_market/$f $n/$(basename $f); done
ERAS=neutral,2000s_a,2000s_b,2000s_c,2000s_d GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${tag}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${tag}_eval.log 2>&1
ERAS=neutral,2000s_a,2000s_b,2000s_c,2000s_d SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${tag}_content.json > logs/${tag}_content.log 2>&1
touch $tag.done
