#!/bin/bash
# E07 paired metrics (av-benchmark): KL-PaSST / KL-PANN of each generated remix against ITS OWN ORIGINAL clip (real mix, same song id) and the LAION-CLAP (music_audioset_epoch_15_esc_90.14) text-audio score of the generated remix against the era prompt it was generated from.
# usage: [EL="1970s 1990s 2000s"] run_avb_pair.sh <gpu> <set,set,...>   (sets as in run_avb.sh; eras 1960s 1980s 2010s neutral). Output: ~/era_market/avbp/out/<set>_<era>.json, flag avbp_<sets>.done. Same environment as run_avb.sh.
cd ~/era_market; export TORCH_HOME=$HOME/era_market/torch_home CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4; source ~/miniconda3/etc/profile.d/conda.sh; conda activate hw3_avbench
AV=/home/jtan/comme5070_pa3/code/av-benchmark; W=~/era_market/avbp; mkdir -p $W/gt $W/pred $W/cache $W/out
for s in ${2//,/ }; do for e in ${EL:-1960s 1980s 2010s} neutral; do
  D=gen_$s; [ $s = gen ] && D=gen; P=$W/pred/${s}_$e; G=$W/gt/${s}_$e; rm -rf $P $G; mkdir -p $P $G
  for f in ~/era_market/$D/*_${e}_s*_remix.wav; do [ -f $f ] || continue; b=$(basename $f .wav); ln -sf $f $P/$b.wav; done
  [ -n "$(ls $P)" ] || { echo "no files for $s $e" >&2; continue; }
  ./venv/bin/python - <<PY
import sys, os, csv, re, ast; sys.path.insert(0, "src")
from common import manifest
import prompts_era; cap = prompts_era.ERAS["$e"]   # same texts as 07_generate.py (checked: 48_lora_musicgen.py and 61_triple_guidance.py use these)
mp = {r["sample_id"]: r["path"] for r in manifest("A")}; rows = []
for f in sorted(os.listdir("$P")):
    b = f[:-4]; sid = b.split("_${e}_")[0]; os.symlink(mp[sid], "$G/" + b + ".wav"); rows.append((b, cap))
csv.writer(open("$W/${s}_${e}.csv", "w")).writerows([("name", "caption")] + rows)
PY
  rm -rf $W/cache/gt_${s}_$e $W/cache/pred_${s}_$e
  ( cd $AV; python extract_text.py --text_csv $W/${s}_${e}.csv --output_cache_path $W/cache/gt_${s}_$e > $W/out/${s}_${e}.log 2>&1
    python evaluate.py --gt_audio $G --gt_cache $W/cache/gt_${s}_$e --pred_audio $P --pred_cache $W/cache/pred_${s}_$e --audio_length=10 --skip_video_related --recompute_pred_cache --num_workers 4 >> $W/out/${s}_${e}.log 2>&1 && cp $W/cache/pred_${s}_$e/output_metrics.json $W/out/${s}_${e}.json )
done; done
touch avbp_$2.done
