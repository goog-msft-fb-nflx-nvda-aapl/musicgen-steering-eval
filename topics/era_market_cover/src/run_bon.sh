#!/bin/bash
# server: best-of-N (G09): 8 candidates (seeds 10..17) for 1960s/1980s/2010s prompts on the 24 songs, then scoring. usage: run_bon.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
S="10 11 12 13 14 15 16 17"; SC=10,11,12,13,14,15,16,17
GENDIR=gen_bon ./venv/bin/python src/07_generate.py --n 24 --eras 1960s 1980s 2010s --seeds $S > logs/bon_gen.log 2>&1
ERAS=1960s,1980s,2010s GENDIR=gen_bon ./venv/bin/python src/10_eval_gen.py --out results/bon_eval.json --train_cond acccodec_norm --seeds $S --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/bon_eval.log 2>&1
ERAS=1960s,1980s,2010s SEEDS=$SC GENDIR=gen_bon ./venv/bin/python src/13_content_metrics.py --out results/bon_content.json > logs/bon_content.log 2>&1
./venv/bin/python src/61_lalm_era.py --out results/bon_lalm.json --gens gen_bon --seeds $S > logs/bon_lalm.log 2>&1
./venv/bin/python src/67_bestofn_analysis.py --eval results/bon_eval.json --content results/bon_content.json --lalm results/bon_lalm.json > results/bon_summary.txt 2> logs/bon_analysis.err
touch bon.done
