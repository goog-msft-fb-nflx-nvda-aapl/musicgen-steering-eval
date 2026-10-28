#!/bin/bash
cd ~/era_market; export CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
GENDIR=${G:-gen_lora} LORA=${V:-v1} ./venv/bin/python src/07_generate.py --ds A --eras neutral 1960s 1980s 2010s --n 24 --seeds 0 1 2 > logs/${P:-lora}_gen.log 2>&1
GENDIR=${G:-gen_lora} ./venv/bin/python src/10_eval_gen.py --out results/${P:-lora}_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/${P:-lora}_eval.log 2>&1
SEEDS=0,1,2 GENDIR=${G:-gen_lora} ./venv/bin/python src/13_content_metrics.py --out results/${P:-lora}_content.json > logs/${P:-lora}_content.log 2>&1
touch ${P:-lora}_gen.done
