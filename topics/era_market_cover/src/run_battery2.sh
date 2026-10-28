#!/bin/bash
# evaluator battery for the newer generators: VINet identity, then the LALM judge. usage: run_battery2.sh <gpu>
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
L="gen_lora_ff gen_lora_pd0 gen_lora_pd3 gen_lora_lr3e5 gen_lora_lr3e4 gen_lora_lr3e4s1 gen_lora_lr3e5s1 gen_lora_nomel"
M="mclg_t3 mclg_t12 mclg_c3 mclg_c075 mcl_gen"
./venv/bin/python src/54_vinet_identity.py $L $M > logs/vinet_battery2.log 2>&1
./venv/bin/python src/61_lalm_era.py --out results/lalm_era_b2_lora.json --gens $L gen_g06_qwen3_8b gen_g06_qwen25_7b gen_g06_qwen25_14b gen_g06_llama32_1b gen_delta_g3 gen_delta_g5 > logs/lalm_b2_lora.log 2>&1
./venv/bin/python src/61_lalm_era.py --out results/lalm_era_b2_mcl.json --seeds 0 1 --gens $M > logs/lalm_b2_mcl.log 2>&1
touch battery2.done
