#!/bin/bash
# main-decade (1960s/1980s/2010s) three-branch runs for the seed adapters at ge15/ge20, started after the matching extra-decade runs. usage: q_main3.sh <gpu> <ge> <adapter>
cd ~/era_market; until [ -f trix_lora_m3e$2$3.done ]; do sleep 60; done
bash run_tri_ad.sh $1 lora 3 $2 50 $3
