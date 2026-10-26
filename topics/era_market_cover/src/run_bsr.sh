#!/bin/bash
# server: BS-RoFormer separation of dataset $1 on GPU $2, then MERT + MuQ embeddings of its stems and probes vs the Demucs stems
cd ~/era_market/src; export PATH=$HOME/era_market/bin:$PATH CUDA_VISIBLE_DEVICES=$2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ds=$1
../venv_sep/bin/python 29_separate_bsr.py --ds $ds > ../logs/sep_bsr_$ds.log 2>&1
cd ..; PY=./venv/bin/python
$PY src/02_embed.py --ds $ds --conds bsr_voc_norm bsr_acc_norm > logs/emb_bsr_mert_$ds.log 2>&1
$PY src/02b_embed_muq.py --ds $ds --conds bsr_voc_norm bsr_acc_norm > logs/emb_bsr_muq_$ds.log 2>&1
$PY src/04_probe.py --ds $ds --conds voc_norm acc_norm bsr_voc_norm bsr_acc_norm --out results/probe_bsr_mert_$ds.json > logs/probe_bsr_mert_$ds.log 2>&1
$PY src/04_probe.py --ds $ds --conds voc_norm acc_norm bsr_voc_norm bsr_acc_norm --prefix muq_ --out results/probe_bsr_muq_$ds.json > logs/probe_bsr_muq_$ds.log 2>&1
touch bsr_$ds.done
