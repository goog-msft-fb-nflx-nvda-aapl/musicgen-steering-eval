#!/bin/bash
# server: recompute everything that used MERT after the inv_freq fix (mert_util.py). usage: bash run_rerun_all.sh probes | evalA <gpu> | evalB <gpu>
# waits for reembed_A.done and reembed_B.done (src/run_reembed.sh). old outputs of the buggy loader are kept in results_buggy_mert/.
cd ~/era_market; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; PY=./venv/bin/python; mode=$1
while [ ! -f reembed_A.done ] || [ ! -f reembed_B.done ]; do sleep 30; done
mkdir -p results_buggy_mert; cp -n results/*.json results/*.txt results_buggy_mert/ 2>/dev/null
if [ $mode = probes ]; then
  for ds in A B; do
    $PY src/04_probe.py --ds $ds --out results/probe_$ds.json > logs/probe_$ds.log 2>&1
    $PY src/04_probe.py --ds $ds --conds acc_norm accharm_norm accperc_norm --out results/probe_hpss_mert_$ds.json > logs/probe_hpss_mert_$ds.log 2>&1
    for cs in 1 2; do $PY src/04_probe.py --ds $ds --cv_seed $cs --conds mix_norm voc_norm acc_norm recomb_norm --out results/probe_cvseed${cs}_mert_$ds.json > logs/probe_cvseed${cs}_mert_$ds.log 2>&1; done
    $PY src/06_extra_probes.py --ds $ds --emb ~/era_market/emb --probe results/probe_$ds.json --out results/extra_$ds.json > logs/extra_$ds.log 2>&1
  done
  $PY src/04_probe.py --ds A --conds acc_norm acccodec_norm --out results/probe_codec_mert_A.json > logs/probe_codec_mert.log 2>&1
  for cs in 1 2; do for ds in A B; do $PY src/04_probe.py --ds $ds --cv_seed $cs --prefix muq_ --conds mix_norm voc_norm acc_norm recomb_norm --out results/probe_cvseed${cs}_muq_$ds.json > logs/probe_cvseed${cs}_muq_$ds.log 2>&1; done; done
  for ds in A B; do for pre in "" muq_; do $PY src/25_lws_probe.py --ds $ds --prefix "$pre" --out results/lws_${pre}$ds.json > logs/lws_${pre}$ds.log 2>&1; done; done
  $PY src/22_analysis_metrics.py --out results/analysis_metrics.json > results/analysis_metrics.txt 2> logs/analysis_metrics.err
  touch rerun_probes.done
else
  export CUDA_VISIBLE_DEVICES=$2
  while [ ! -f rerun_probes.done ]; do sleep 30; done   # eval needs the codec probe layer choices
  P="--train_cond acccodec_norm --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json"
  if [ $mode = evalA ]; then
    GENDIR=gen $PY src/10_eval_gen.py --out results/gen_eval_codec_24s3.json $P --seeds 0 1 2 > logs/gen_eval_24s3.log 2>&1
    $PY src/16_controls.py --out results/controls_24s3.json --seeds 0 1 2 > logs/controls_24s3.log 2>&1
    $PY src/20_kad.py results/gen_eval_codec_24s3_emb.npz --out results/kad_24s3.json --layers results/gen_eval_codec_24s3.json > logs/kad_24s3.log 2>&1
    GENDIR=mcl_gen $PY src/10_eval_gen.py --out results/mcl_eval.json $P --seeds 0 1 2 > logs/mcl_eval.log 2>&1
    touch rerun_evalA.done
  else
    for g in 1 5 7; do GENDIR=gen_cfg$g $PY src/10_eval_gen.py --out results/cfg${g}_eval.json $P --seeds 0 > logs/cfg${g}_eval.log 2>&1; done
    GENDIR=gen $PY src/10_eval_gen.py --out results/cfg3_eval.json $P --seeds 0 > logs/cfg3_eval.log 2>&1
    GENDIR=gen_fx $PY src/10_eval_gen.py --out results/fx_eval.json $P --seeds 0 > logs/fx_eval.log 2>&1
    touch rerun_evalB.done
  fi
fi
