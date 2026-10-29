#!/bin/bash
# replicate candidate draws for the LLM prompt writers. usage: run_g06_rep.sh <gpu> <hf id> <base tag> <rep tag suffix> <seed0>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6; T=$3$4
[ -f results/llm_prompts_$T.json ] || SEED0=$5 ./venv/bin/python src/69_llm_prompts.py $2 $T > logs/g06_${T}_prompts.log 2>&1
./run_g06r.sh $1 $2 $T
