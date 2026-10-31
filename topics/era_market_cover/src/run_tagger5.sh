#!/bin/bash
# Discogs-EffNet evaluator (CPU, venv_ess) for the extra-decade sets and the base ge7 main-decade set, one process at a time
cd ~/era_market; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 CUDA_VISIBLE_DEVICES=
for n in trix_lora_m3e7 trix_lora_m3e10 trix_base_m3e7; do
  ./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_$n --eras neutral,1970s,1990s,2000s --out results/discogs_$n.json > logs/discogs_$n.log 2>&1
done
./venv_ess/bin/python src/36_discogs_tagger.py --gen gen_tri_base_m3e7 --out results/discogs_tri_base_m3e7.json > logs/discogs_tri_base_m3e7.log 2>&1
touch tagger5.done
