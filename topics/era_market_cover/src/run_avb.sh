#!/bin/bash
# E07: FD-VGG / FD-PANN / FD-PaSST (av-benchmark, unpaired) of generated sets against real PA1 train clips of the target decade.
# usage: run_avb.sh <gpu> <set,set,...>   sets are gen_<name> dirs (or `gen` for the standard base set); eras 1960s 1980s 2010s and neutral. References: dataset A train clips per decade (171 each).
# av-benchmark checkout and env are the ones installed in ~/comme5070_pa3 (read-only use); caches/outputs live in ~/era_market/avb.
cd ~/era_market; export TORCH_HOME=$HOME/era_market/torch_home CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4; source ~/miniconda3/etc/profile.d/conda.sh; conda activate hw3_avbench
AV=/home/jtan/comme5070_pa3/code/av-benchmark; W=~/era_market/avb; mkdir -p $W/ref $W/pred $W/cache $W/out
./venv/bin/python - <<P
import sys, os; sys.path.insert(0, "src")
from common import manifest
for r in manifest("A"):
    if r["split"] == "train":
        d = f"$W/ref/{r['label']}"; os.makedirs(d, exist_ok=True); l = f"{d}/{r['sample_id']}.wav"
        if not os.path.exists(l): os.symlink(r["path"], l)
P
for s in ${2//,/ }; do for e in 1960s 1980s 2010s neutral; do
  P=$W/pred/${s}_$e; mkdir -p $P; D=gen_$s; [ $s = gen ] && D=gen   # the standard-CFG base set lives in gen/
  for f in ~/era_market/$D/*_${e}_s*_remix.wav; do [ -f $f ] && ln -sf $f $P/$(basename $f); done
  [ -n "$(ls $P 2>/dev/null)" ] || { echo "no files for $s $e" >&2; continue; }
  for d in 1960s 1980s 2010s; do [ $e = neutral ] || [ $e = $d ] || continue
    [ -f $W/out/${s}_${e}_vs_$d.json ] && continue
    ( cd $AV; python evaluate.py --gt_audio $W/ref/$d --gt_cache $W/cache/gt_$d --pred_audio $P --pred_cache $W/cache/pred_${s}_$e --audio_length=10 --unpaired --skip_video_related --recompute_pred_cache --num_workers 4 > $W/out/${s}_${e}_vs_$d.log 2>&1 && cp $W/cache/pred_${s}_$e/output_metrics.json $W/out/${s}_${e}_vs_$d.json )
  done
done; done
touch avb_$2.done
