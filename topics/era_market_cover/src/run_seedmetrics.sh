#!/bin/bash
# usage: run_seedmetrics.sh <gpu> <suffix list e.g. "m3e15v1s1 m3e10v1s1"> <extra trix-only sets>
# FD-VGG (unpaired) and paired metrics for seed-adapter sets; main decades (tri_) and extra decades (trix_). Skips outputs that exist.
cd ~/era_market; g=$1
for k in $2; do
  [ -f avb/out/tri_lora_${k}_neutral_vs_1960s.json ] || bash run_avb.sh $g tri_lora_$k
  [ -f avbp/out/tri_lora_${k}_neutral.json ] || bash run_avb_pair.sh $g tri_lora_$k
done
for k in $2 $3; do
  [ -f avb/out/trix_lora_${k}_neutral_vs_1970s.json ] || EL="1970s 1990s 2000s" bash run_avb.sh $g trix_lora_$k
  [ -f avbp/out/trix_lora_${k}_neutral.json ] || EL="1970s 1990s 2000s" bash run_avb_pair.sh $g trix_lora_$k
done
touch seedmetrics_$g.done
