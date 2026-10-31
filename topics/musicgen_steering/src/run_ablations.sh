#!/bin/bash
# server-side launcher for the 2026-10-19 ablation round. usage: bash run_ablations.sh <gpu> <which>
#   which = medium_hi : medium, eta .9/1.2/1.6/2.0, meandiff then rfm        (out arms_medium_hi/)
#   which = comp      : small, rfm eigendirection index 1 and 2 (const8/musicrfm arms) (out arms_comp1/, arms_comp2/)
#   which = conf      : small, 10b_second_eval.py rerun (adds chroma argmax per clip) for meandiff and rfm -> arms_eval2b/ (target -> predicted note confusion, B/D# diagnosis)
#   which = seeds     : small, rfm + meandiff directions retrained with seeds 43 and 44 and re-steered (out arms_seed43/, arms_seed44/)
source ~/miniconda3/bin/activate musicgen_steer; cd ~/musicgen_steer
export HF_HOME=~/musicgen_steer/hf CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
NOTES=(C C# D D# E F F# G G# A A# B)
case "$2" in
medium_hi)
  for m in meandiff rfm; do for n in "${NOTES[@]}"; do
    python 03_arm_compare_v2.py --model medium --method $m --note "$n" --etas 0.9 1.2 1.6 2.0 --gen_seeds 0 1 --out ./arms_medium_hi > hi_${m}_${n//#/sharp}.log 2>&1
  done; done ;;
comp)
  for c in 1 2; do for n in "${NOTES[@]}"; do
    python 03_arm_compare_v2.py --model small --method rfm --comp $c --note "$n" --gen_seeds 0 1 --out ./arms_comp$c > comp${c}_${n//#/sharp}.log 2>&1
  done; done ;;
seeds)
  for s in 43 44; do
    for n in "${NOTES[@]}"; do python 01_train_note_direction.py --model small --note "$n" --seed $s > train_s${s}_${n//#/sharp}.log 2>&1; done
    CUDA_VISIBLE_DEVICES= python 06_direction_methods.py --model small --seed $s --suffix _s$s > alt_s$s.log 2>&1
    for m in rfm meandiff; do for n in "${NOTES[@]}"; do
      python 03_arm_compare_v2.py --model small --method $m --note "$n" --train_seed $s --alt_suffix _s$s --gen_seeds 0 1 --out ./arms_seed$s > seed${s}_${m}_${n//#/sharp}.log 2>&1
    done; done
  done ;;
conf)
  for m in meandiff rfm; do for n in "${NOTES[@]}"; do
    python 10b_second_eval.py --model small --method $m --note "$n" --gen_seeds 0 1 --out ./arms_eval2b > conf_${m}_${n//#/sharp}.log 2>&1
  done; done ;;
esac
echo finished > ablation_$2.done
