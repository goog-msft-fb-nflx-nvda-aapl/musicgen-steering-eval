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

## 2026-10-14 pilot: D# steering on musicgen-small (first look, not the full pilot)
config: src/02_pilot_note_steer.py --model small --note D#; direction = results/note_probe/ D# direction (n_components 12, rfm_iters 30, train seed 42), unit-norm top direction per layer (norm 1.0). steering layers = top-8 by val score: -13,-12,-11,-9,-8,-7,-6,-5. injection: add eta * direction to each chosen decoder layer's output at every step (constant, no layer weights, no inject_chance). generation: guidance_scale 3.0, sampling, 500 tokens (10 s), 20 text prompts (listed in the json) x 2 generation seeds, same torch seed for every eta so baseline and steered are paired, batch of 20. eta in {0, 0.3, 0.6, 1.0, 1.5}. evaluator: librosa chroma_cqt, time-mean, hit = argmax pitch class == D#; frac = D# share of total chroma energy. gpu 2 (shared), OMP/MKL 4 threads. raw rows + 3 flac demos per (seed, eta) in results/pilot/.

measured, 40 paired clips per eta (20 prompts x 2 seeds), vs eta=0:
- eta 0.3: hit 0.025 -> 0.275 (11 discordant up, 1 down), exact McNemar p=0.0063; frac 0.074 -> 0.130, Wilcoxon p<1e-4
- eta 0.6: hit 0.025 -> 0.125 (5 up, 1 down), p=0.219; frac 0.074 -> 0.099, Wilcoxon p=0.0018
- eta 1.0: hit 0.025 -> 0.000, p=1.0; frac 0.074 -> 0.062, p=0.44
- eta 1.5: hit 0.025 -> 0.000, p=1.0; frac 0.074 -> 0.060, p=0.31
- uniform chroma would give frac 0.083 and entropy 3.58 bits; mean chroma entropy by eta: 3.335 (0), 3.383 (0.3), 3.446 (0.6), 3.241 (1.0), 3.311 (1.5).

caveats: 40 pairs are not independent (20 prompts reused across the 2 seeds); p-values are uncorrected across the 4 eta comparisons (not Holm-adjusted); one note, one model size, one layer-selection rule, one evaluator. the chroma-argmax hit is a weak evaluator (time-mean chroma of polyphonic music, no check that D# is the tonic or a salient note); not yet validated on real labeled audio. i have not listened to the clips, and no audio-quality metric (clap/mad) has been computed, so why the eta>=1.0 rows go to 0 hits is not established. interpretation only: the non-monotonic pattern (up at 0.3, weaker at 0.6, none at >=1.0) is what the control-vs-quality tradeoff in PLAN Q2 would need to look at.

infra note: transformers 5.18 returns a tensor from decoder layers (not a tuple), which breaks musicrfm.generation_utils.hook_model (output[0] indexes the batch); the pilot uses its own add_hooks. any future generation code must not use the library hook.
