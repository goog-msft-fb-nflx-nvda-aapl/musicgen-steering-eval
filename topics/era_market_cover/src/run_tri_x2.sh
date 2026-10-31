#!/bin/bash
# three-branch guidance for the extra decades with adapter choice and content metrics. usage: run_tri_x2.sh <gpu> <gm> <ge> [adapter under lora/, default v1]
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
gm=$2; ge=$3; ad=${4:-v1}; n=gen_trix_lora_m${gm}e${ge}${ad}; ns=gen_lora; [ $ad = v1 ] || ns=gen_lora_$ad; export LORA=$ad NEUTRAL_SRC=$ns; tag=trix_lora_m${gm}e${ge}${ad}
./venv/bin/python src/61_triple_guidance.py --gm $gm --ge $ge --eras 1970s 1990s 2000s --outdir $n --seeds 0 1 2 > logs/$tag.log 2>&1
for f in $ns/*_neutral_s[012]_acc.wav; do ln -sf ~/era_market/$f $n/$(basename $f); done
E=neutral,1970s,1990s,2000s
ERAS=$E GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/${tag}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${tag}_eval.log 2>&1
for f in $ns/*_neutral_s[012]_remix.wav; do ln -sf ~/era_market/$f $n/$(basename $f); done
ERAS=$E SEEDS=0,1,2 GENDIR=$n ./venv/bin/python src/13_content_metrics.py --out results/${tag}_content.json > logs/${tag}_content.log 2>&1
touch ${tag}.done
