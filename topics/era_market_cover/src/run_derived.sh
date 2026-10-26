#!/bin/bash
# server: HPSS probes (accharm_norm / accperc_norm, MERT + MuQ), parametric production-only control (gen_fx), content metrics with 3 seeds
cd ~/era_market; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY=./venv/bin/python
$PY src/23_derive.py hpss > logs/derive_hpss.log 2>&1
$PY src/23_derive.py fx > logs/derive_fx.log 2>&1
export CUDA_VISIBLE_DEVICES=$1
for ds in A B; do
  $PY src/02_embed.py --ds $ds --conds accharm_norm accperc_norm > logs/emb_hpss_mert_$ds.log 2>&1
  $PY src/02b_embed_muq.py --ds $ds --conds accharm_norm accperc_norm > logs/emb_hpss_muq_$ds.log 2>&1
  $PY src/04_probe.py --ds $ds --conds acc_norm accharm_norm accperc_norm --out results/probe_hpss_mert_$ds.json > logs/probe_hpss_mert_$ds.log 2>&1
  $PY src/04_probe.py --ds $ds --conds acc_norm accharm_norm accperc_norm --prefix muq_ --out results/probe_hpss_muq_$ds.json > logs/probe_hpss_muq_$ds.log 2>&1
done
GENDIR=gen_fx $PY src/10_eval_gen.py --out results/fx_eval.json --train_cond acccodec_norm --seeds 0 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/fx_eval.log 2>&1
GENDIR=gen_fx SEEDS=0 $PY src/13_content_metrics.py --out results/fx_content.json > logs/fx_content.log 2>&1
SEEDS=0,1,2 $PY src/13_content_metrics.py --out results/content_metrics_24s3.json > logs/content24s3.log 2>&1
touch derived.done
