#!/bin/bash
# server: score MuseControlLite generations (mcl_gen, seeds 0 1 2) with the same metrics as MusicGen: codec-matched era probes, content, CLAP zero-shot era, COCOLA
cd ~/era_market; export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 GENDIR=mcl_gen
./venv/bin/python src/10_eval_gen.py --out results/mcl_eval.json --train_cond acccodec_norm --seeds 0 1 2 --probe_mert probe_codec_mert_A.json --probe_muq probe_codec_muq_A.json > logs/mcl_eval.log 2>&1
SEEDS=0,1,2 ./venv/bin/python src/13_content_metrics.py --out results/mcl_content.json > logs/mcl_content.log 2>&1
sed 's|G = os.path.join(ROOT, "gen")|G = os.path.join(ROOT, "mcl_gen")|' src/26_clap_era.py > src/26_clap_era_mcl.py
./venv_mcl/bin/python src/26_clap_era_mcl.py --out results/mcl_clap_era.json > logs/mcl_clap.log 2>&1
sed 's|ap.add_argument("--gen", default="gen")|ap.add_argument("--gen", default="mcl_gen")|' src/24_cocola.py > src/24_cocola_mcl.py
cd ext/cocola; source ../../venv_ext/bin/activate; python ../../src/24_cocola_mcl.py --out ../../results/mcl_cocola.json > ../../logs/mcl_cocola.log 2>&1
touch ~/era_market/mcl_eval.done
