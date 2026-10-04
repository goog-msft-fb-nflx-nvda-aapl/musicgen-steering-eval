# worklog

## 2026-10-05
- picked topic from docs/response.md (see PLAN.md for scope changes).
- repo + docs set up. nothing run yet.

## 2026-10-05 (server setup)
- gsm-gpu2 nvidia-smi: gpu 0/2/3 idle, gpu 1 had 9.3 GB used by something else. used gpu 2 (real cuda matmul ok, H200 NVL 143 GB).
- base conda torch is too new for the driver (cuda 12.4 max). new env `musicgen_steer` (py3.11, torch 2.5.1+cu124). music-rfm deps installed with pip --no-deps -e so torch stays pinned. transformers 5.18.0, xrfm installed.
- music-rfm cloned at ~/musicgen_steer/music-rfm (commit f2a8d96). server has no github credentials for the private repo; syncing src via scp for now.
- smoke (src/smoke_gen.py): musicgen-small, hf transformers, sampling, 500 tokens (~10 s clip), gpu 2, OMP_NUM_THREADS=4. bs=1 2.90 s/clip, bs=8 0.58 s/clip, peak 5.7 GB. at that rate ~4800 clips is ~47 min of small generation (generation only, hooks/eval not included). large not measured yet.
- transformers warns pad/bos token id 2048 outside vocab 0-2047 in the musicgen config; not investigated yet.
- the music-rfm examples train on all layers (-1..-47 for large); small has 24 decoder layers. quickstart (syntheory notes) not run yet.
- rule from user: one process at a time on the server.
