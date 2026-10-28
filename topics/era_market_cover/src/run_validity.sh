#!/bin/bash
# server: independent evaluators (Audiobox, KAD-style margin, CLAP zero-shot era, COCOLA) for LoRA v1/v2/v3 and ACE-Step c05/c10 generations; Discogs tagger for v1 and ACE c05
cd ~/era_market; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for spec in "lora gen_lora lora" "lora_v2 gen_lora_v2 lora_v2" "lora_v3 gen_lora_v3 lora_v3" "ace_c05 ace_gen_c05 ace_c05" "ace_c10 ace_gen_c10 ace_c10"; do
  set -- $spec; ev=$1; d=$2; n=$3
  GENDIR=$d ./venv2/bin/python src/18_quality.py --seeds 0 1 2 --out results/quality_$n.json > logs/quality_$n.log 2>&1
  ./venv/bin/python src/20_kad.py results/${ev}_eval_emb.npz --out results/kad_$n.json --layers results/${ev}_eval.json > logs/kad_$n.log 2>&1
  GENDIR=$d ./venv_mcl/bin/python src/26_clap_era_env.py --out results/clap_era_$n.json > logs/clap_$n.log 2>&1
  (cd ext/cocola; source ../../venv_ext/bin/activate; python ../../src/24_cocola.py --gen $d --out ../../results/cocola_$n.json > ../../logs/cocola_$n.log 2>&1)
done
touch validity.done
