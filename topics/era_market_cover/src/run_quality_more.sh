#!/bin/bash
# server: Audiobox Aesthetics + KAD-style margin for the guidance sweep, MuseControlLite and text-only generations. usage: bash run_quality_more.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for spec in "gen_cfg1 0 cfg1" "gen_cfg5 0 cfg5" "gen_cfg7 0 cfg7" "mcl_gen 0 1 2 mcl" "gen_textonly 0 1 2 textonly"; do
  set -- $spec; d=$1; shift; name=${@: -1}; seeds=${@:1:$#-1}
  GENDIR=$d ./venv2/bin/python src/18_quality.py --seeds $seeds --out results/quality_$name.json > logs/quality_$name.log 2>&1
done
for name in cfg1 cfg5 cfg7 mcl textonly; do
  ./venv/bin/python src/20_kad.py results/${name}_eval_emb.npz --out results/kad_$name.json --layers results/${name}_eval.json > logs/kad_$name.log 2>&1
done
touch quality_more.done
