#!/bin/bash
# G08b: MuseControlLite guidance sweep. usage: run_mcl_g.sh <gpu> <name> <gtext> <gcon> [gaudio]   (repo defaults 7.0 / 1.5 / 1.0); 24 songs, 4 prompts, seeds 0 1
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; n=$2
(cd ext/MuseControlLite && ~/era_market/venv_mcl/bin/python ~/era_market/src/27_musecontrol.py --seeds 0 1 --outdir mclg_$n --gtext $3 --gcon $4 ${5:+--gaudio $5}) > logs/mclg_${n}_gen.log 2>&1
GENDIR=mclg_$n ./venv/bin/python src/10_eval_gen.py --out results/mclg_${n}_eval.json --train_cond acccodec_norm --seeds 0 1 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/mclg_${n}_eval.log 2>&1
SEEDS=0,1 GENDIR=mclg_$n ./venv/bin/python src/13_content_metrics.py --out results/mclg_${n}_content.json > logs/mclg_${n}_content.log 2>&1
touch mclg_$n.done
