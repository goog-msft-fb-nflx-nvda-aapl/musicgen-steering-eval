#!/bin/bash
# R03: same seed, same inputs, twice on GPU 2 and once on GPU 3; compare outputs. usage: run_det.sh
cd ~/era_market; export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for v in "a 2" "b 2" "c 3"; do set -- $v; CUDA_VISIBLE_DEVICES=$2 GENDIR=gen_det_$1 ./venv/bin/python src/07_generate.py --ds A --eras neutral 1980s --n 6 --seeds 0 > logs/det_$1.log 2>&1; done
touch det.done
