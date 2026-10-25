#!/bin/bash
# server: waits for stems_emb.done, then stats, language ID, probes (probes are CPU)
cd ~/era_market; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while [ ! -f stems_emb.done ]; do sleep 30; done
for ds in A B; do ./venv/bin/python src/03_stats.py --ds $ds > logs/stats_$ds.log 2>&1; done
for ds in A B; do CUDA_VISIBLE_DEVICES=$1 ./venv/bin/python src/05_langid.py --ds $ds > logs/lid_$ds.log 2>&1; done
for ds in A B; do ./venv/bin/python src/04_probe.py --ds $ds --out results/probe_$ds.json > logs/probe_$ds.log 2>&1; done
touch analysis.done
