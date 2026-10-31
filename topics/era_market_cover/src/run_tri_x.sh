#!/bin/bash
# three-branch guidance for the extra decades (1970s, 1990s, 2000s). usage: run_tri_x.sh <gpu> <base|lora> <gm> <ge>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
m=$2; gm=$3; ge=$4; n=gen_trix_${m}_m${gm}e${ge}; src=gen; ns=gen
if [ $m = lora ]; then export LORA=v1 NEUTRAL_SRC=gen_lora; ns=gen_lora; fi
./venv/bin/python src/61_triple_guidance.py --gm $gm --ge $ge --eras 1970s 1990s 2000s --outdir $n --seeds 0 1 2 > logs/trix_${m}_m${gm}e${ge}.log 2>&1
for f in $ns/*_neutral_s[012]_acc.wav; do ln -sf ~/era_market/$f $n/$(basename $f); done
ERAS=neutral,1970s,1990s,2000s GENDIR=$n ./venv/bin/python src/10_eval_gen.py --out results/trix_${m}_m${gm}e${ge}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/trix_${m}_m${gm}e${ge}_eval.log 2>&1
touch trix_${m}_m${gm}e${ge}.done
