#!/bin/bash
# server: EnCodec-resynthesized real accompaniment (dataset A) -> MERT/MuQ embeddings -> probes (acc_norm vs acccodec_norm)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
./venv/bin/python src/12_codec_resynth.py --ds A > logs/codec_A.log 2>&1
./venv/bin/python src/02_embed.py --ds A --conds acccodec_norm > logs/emb_codec_mert.log 2>&1
./venv/bin/python src/02b_embed_muq.py --ds A --conds acccodec_norm > logs/emb_codec_muq.log 2>&1
./venv/bin/python src/04_probe.py --ds A --conds acc_norm acccodec_norm --out results/probe_codec_mert_A.json > logs/probe_codec_mert.log 2>&1
./venv/bin/python src/04_probe.py --ds A --conds acc_norm acccodec_norm --prefix muq_ --out results/probe_codec_muq_A.json > logs/probe_codec_muq.log 2>&1
touch codec.done
