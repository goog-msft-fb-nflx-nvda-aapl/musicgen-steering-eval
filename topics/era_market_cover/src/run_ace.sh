#!/bin/bash
# server: ACE-Step 1.5 cover mode on the original accompaniment stem, three cover strengths, then the standard scoring
cd ~/era_market; export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for c in 02 05 08; do
  s=$(python3 -c "print(int(\"$c\")/10)")
  (cd ext/ACE-Step-1.5; CUDA_VISIBLE_DEVICES=2 ~/era_market/venv_ace/bin/python ../../src/55_ace_cover.py --strength $s --outdir ace_gen_c$c --seeds 0 1 2 > ../../logs/ace_gen_c$c.log 2>&1)
  CUDA_VISIBLE_DEVICES=2 GENDIR=ace_gen_c$c ./venv/bin/python src/10_eval_gen.py --out results/ace_c${c}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/ace_c${c}_eval.log 2>&1
  SEEDS=0,1,2 GENDIR=ace_gen_c$c ./venv/bin/python src/13_content_metrics.py --out results/ace_c${c}_content.json > logs/ace_c${c}_content.log 2>&1
done
touch ace.done
