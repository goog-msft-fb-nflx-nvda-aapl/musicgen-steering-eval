#!/bin/bash
cd ~/era_market; export OMP_NUM_THREADS=4; PY=./venv/bin/python
$PY src/38_remix_level.py > logs/remix_level.log 2>&1
for t in m12 m6 p0 p6; do GENDIR=gen_lvl_$t SEEDS=0 $PY src/13_content_metrics.py --out results/remixlvl_$t.json > logs/remixlvl_$t.log 2>&1; done
touch remixlvl.done
