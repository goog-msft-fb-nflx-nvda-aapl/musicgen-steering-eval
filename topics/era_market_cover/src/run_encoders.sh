cd ~/era_market/src
for e in whisper ast clap; do for d in A B; do CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 ~/era_market/venv/bin/python 46_embed_more.py --ds $d --enc $e >> ../logs/enc_$e.log 2>&1; done; done
touch ../encoders.done
