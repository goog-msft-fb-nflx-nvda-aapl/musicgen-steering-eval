#!/bin/bash
# Audiobox Aesthetics (18_quality.py) + KAD-style margin (20_kad.py) for generation sets that had neither. usage: run_qk.sh <gpu> <tag> <name:seeds,...>  (dir = gen_<name>; embeddings from results/<name>_eval_emb.npz)
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; tag=$2
for spec in ${3//,/ }; do
  name=${spec%%:*}; seeds=${spec#*:}; seeds=${seeds//-/ }
  GENDIR=gen_$name ./venv2/bin/python src/18_quality.py --seeds $seeds --out results/quality_$name.json > logs/quality_$name.log 2>&1
  ./venv/bin/python src/20_kad.py results/${name}_eval_emb.npz --out results/kad_$name.json --layers results/${name}_eval.json > logs/kad_$name.log 2>&1
done
touch qk_$tag.done
