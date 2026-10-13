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

## 2026-10-14 first experiment: note-D# probe/direction on musicgen-small
config: src/01_train_note_direction.py --model small --note D# (n_components 12, rfm_iters 30, seed 42, hyperparam_samples 100, tuning_metric auc, pooling mean). data: syntheory "notes", D# vs all other notes, balanced 1 pos : 1 neg, random 70/15/15 split of clips (train 1158 / val 248 / test 250 = 125 pos + 125 neg). features: encodec_32khz codes of the clip fed to musicgen-small decoder, mean-pooled hidden state per layer, layers -1..-23 (24-layer decoder). gpu 2 (shared with other users' jobs), OMP/MKL 4 threads, hf cache ~/musicgen_steer/hf. result file: results/note_probe/musicgen_small_note_Dsharp_ncomp12_seed42.json.

measured (test set, n=250, accuracy %):
- rfm predictor: range 87.2 (layer -16) to 94.4 (layer -7); 94.0 at -6 and -13; 93.6 at -5.
- top-direction (the vector steering would inject): range 84.4 (-17) to 93.2 (-5, -8); 'aggregated' direction acc 88.8, auc 0.888.
- layers ranked by val score (the criterion music-rfm uses to pick steering layers): -12 (0.944), -8 (0.935), -7, -9, -11.
- n=250 gives a binomial SE of about 2 points at ~90%, so differences of a few points between neighbouring layers are not resolved by this run. single seed, single note, single split.

problems met (fixed, patches/music-rfm-num-layers.patch):
- music-rfm hardcodes num_layers = 48 in get_hidden_states_music, KeyError on small. now reads model.config.num_hidden_layers (same value for large). generation hooks not yet checked for the same assumption.
- server has no ffmpeg/torchcodec, datasets audio decode failed. decode wav bytes with soundfile instead.

things i do not trust yet:
- the 'aggregated' precision/recall/f1 in the result json come out >100 (e.g. precision 125.0); that metric code looks wrong, not using those.
- log shows "Tree has no split, stopping training" and "0 iterations per tree" for some layer fits, i.e. some xRFM fits may have done no feature learning. not investigated; need to check how many layers and whether the direction is then just a plain kernel/linear solution.
- classification warning "Using floating point y".

next: pilot generation (steer musicgen-small with this D# direction at eta 0/0.3/0.6, 20 prompts, paired seeds, chroma-argmax evaluator) and fix the generation-side layer assumptions first.
