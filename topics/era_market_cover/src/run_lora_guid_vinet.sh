#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4
until [ -f lorav1_neg_g3.done ] && [ -f lorav1_delta_g3.done ]; do sleep 60; done
./venv/bin/python src/54_vinet_identity.py gen_lorav1_neg_g3 gen_lorav1_delta_g3 > logs/lorav1_vinet.log 2>&1
touch lorav1_vinet.done
