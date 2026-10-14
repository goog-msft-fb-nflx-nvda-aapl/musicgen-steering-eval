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

## 2026-10-14 literature check + next jobs
- survey in SURVEY.md; PLAN.md updated (musicrfm already has chroma/essentia/onset checks, so our novelty is paired stats + fidelity-matched direction-method comparison + decodability vs steerability on music, not "their eval is circular").
- launched on gsm-gpu2 gpu 2, tmux ms:train12: sequential training of note directions C, C#, D, E, F, F#, G, G#, A, A#, B (musicgen-small, same config as D#), logs ~/musicgen_steer/note_small_<note>.log. one process at a time.

## 2026-10-14 queued: arm comparison (D#, musicgen-small)
src/03_arm_compare.py queued in tmux ms:arms; it waits for the train12 window to close, then runs once on gpu 2 (log ~/musicgen_steer/arms_small_Dsharp.log, outputs ~/musicgen_steer/arms/). arms: const8 (top-8 by val score, weight 1, p=1) vs musicrfm-style (all 23 layers, weight = minmax(val score)^(1/0.95), per-layer per-step injection p=0.3, own re-implementation of the paper's description, not their code path). eta in {0.15, 0.3, 0.45, 0.6}, 20 prompts x 4 generation seeds = 80 paired clips per cell, same torch+python rng seed per (seed, eta) so every arm pairs with the eta=0 baseline. evaluator: chroma argmax + D# chroma share. analysis to do on the mac from the json: exact McNemar per cell, Wilcoxon on share, Holm across cells, arm-vs-arm paired comparison.

## 2026-10-14 moved jobs to GPU 1
rule from user: do not share a GPU with other users' processes; use a free GPU. gpu 2 had another user's job (876 MiB), gpu 3 another user's (and gpu 3 is on the broken list), gpu 0 broken. gpu 1 was idle, passed a real matmul check. stopped my own windows (train12, arms) and relaunched on CUDA_VISIBLE_DEVICES=1: tmux ms:train7 = notes F, F#, G, G#, A, A#, B (C, C#, D, D#, E already trained; F was interrupted mid-run and restarted), then ms:arms (03_arm_compare.py, D#) waits for train7 to close. my process on gpu 1: pid 4008731 (5.5 GB at last check). check all gpu indices again before every new launch.

## 2026-10-14 all 12 note directions trained (musicgen-small)
config identical to the D# run above (src/01_train_note_direction.py --model small, n_components 12, rfm_iters 30, seed 42, per-note balanced 1:1 split 70/15/15, test n=250, layers -1..-23, gpu 2 for C..E and part of F, gpu 1 for F..B after the move). result jsons in results/note_probe/ (one per note).
per-note test accuracy over layers, max/min across the 23 layers (%): predictor | steering direction; top-3 layers by val score.
D# 94.4/87.2 | 93.2/84.4 | -12,-8,-7
A# 94.8/84.8 | 93.6/84.0 | -8,-9,-12
A 93.2/86.4 | 94.0/85.2 | -9,-10,-6
B 93.6/82.8 | 94.0/81.6 | -6,-5,-7
C# 93.2/85.2 | 93.2/86.0 | -7,-11,-8
C 96.0/86.4 | 96.8/83.6 | -9,-10,-6
D 94.4/83.6 | 92.8/84.8 | -8,-10,-7
E 94.0/86.8 | 94.0/84.4 | -13,-7,-12
F# 92.4/84.8 | 93.6/84.4 | -13,-7,-5
F 93.6/86.4 | 91.6/78.8 | -6,-5,-7
G# 94.8/84.8 | 95.2/82.0 | -8,-7,-5
G 95.2/87.2 | 95.6/83.6 | -3,-4,-12
single seed, single split per note, n=250 (SE ~2 points), so layer-to-layer and note-to-note differences of a few points are not resolved. the max over 23 layers is also selected on the test numbers shown, so it is optimistic as a summary. top-3 val layers cluster in -5..-13 for most notes (G is the exception with -3,-4).
arm comparison (D#) is running on gpu 1 (pid 111449 at 20:22); its results are not in the repo yet.

## 2026-10-15 arm comparison, D#, musicgen-small (src/03_arm_compare.py, ran 2026-10-14 ~5 min on gpu 1)
config: direction = D# note direction (n_components 12, rfm_iters 30, train seed 42; see 2026-10-14 entries). arms: const8 = top-8 layers by val score, weight 1, inject every step; musicrfm = all 23 layers, weight = minmax(val score)^(1/0.95) (weights: -12 1.0, -8 0.88, -7/-9/-11/-13 0.82, -5/-6/-10 0.75, -4 0.63, -14..-17 0.0, -18..-21 0.16, -1/-2 0.22, -3 0.28, -22 0.11, -23 0.05), per-layer per-step injection p=0.3 (own re-implementation of the paper's description). eta in {0.15, 0.3, 0.45, 0.6}. 20 prompts x 4 generation seeds = 80 paired clips per cell; the same torch+python seed per (seed, eta) so every cell pairs with the unsteered baseline. guidance_scale 3.0, 500 tokens (10 s). evaluator: librosa chroma_cqt time-mean; hit = D# is the argmax pitch class; frac = D# share of chroma energy. gpu 1. raw rows + 27 demo flacs: results/arms/.
stats: exact McNemar on hit, Wilcoxon signed-rank on frac, Holm across the 8 arm x eta cells vs baseline, Holm across 4 etas for arm-vs-arm.

unsteered baseline: hit 1/80 (0.0125), frac 0.068.
vs baseline (hit rate; discordant up/down; McNemar p, Holm p | frac; Holm p):
- const8 0.15: 0.075 (6/1) p=.125, holm .25 | 0.093, holm .0011
- const8 0.30: 0.287 (23/1) p<1e-4, holm <1e-4 | 0.122, holm <1e-4
- const8 0.45: 0.175 (14/1) p=.001, holm .0039 | 0.113, holm <1e-4
- const8 0.60: 0.113 (9/1) p=.0215, holm .0645 | 0.098, holm <1e-4
- musicrfm 0.15: 0.037 (3/1) p=.625, holm .625 | 0.071, holm .55
- musicrfm 0.30: 0.188 (14/0) p=1e-4, holm .0006 | 0.101, holm <1e-4
- musicrfm 0.45: 0.263 (20/0) p<1e-4, holm <1e-4 | 0.118, holm <1e-4
- musicrfm 0.60: 0.475 (37/0) p<1e-4, holm <1e-4 | 0.158, holm <1e-4
arm vs arm (musicrfm - const8), same eta, paired: eta 0.15 p=.45 (holm .55); 0.30 p=.18 (holm .55; const8 ahead, 18 vs 10 discordant); 0.45 p=.26 (holm .55; rfm ahead 18 vs 11); 0.60 p<1e-4 (holm <1e-4; musicrfm-only hit 31, const8-only hit 2).
chroma entropy (bits; uniform 3.585): baseline 3.360, const8 3.375-3.420, musicrfm 3.348-3.393, i.e. no flattening of the chroma at these etas.

observations / interpretation (labelled as such):
- both configs raise D# prominence over baseline in the paired design. const8 hit rate peaks at eta 0.3 and drops at 0.45 and 0.6; the musicrfm-style config keeps rising through 0.6 (0.475). only the 0.6 arm difference survives Holm.
- interpretation only, untested: const8 adds eta to 8 layers at every step (about 8*eta per step in total), the musicrfm-style config adds weighted eta with p=0.3 (expected sum of weights about 11.6*0.3 = 3.5 eta per step). the two configs differ in total injected magnitude as well as in layer weighting and injection prob, so this comparison does not isolate which factor matters; a matched-magnitude comparison (plan Q2: pareto front) is needed.
- the earlier pilot (const8-like, p=1) fell to 0 hits at eta>=1.0; etas above 0.6 were not run here, so the musicrfm-style arm's behaviour beyond 0.6 is unknown.
- caveats: one note, one model size, chroma-argmax evaluator not validated on labelled audio, no fidelity metric (clap/mad) yet, 20 prompts reused across seeds so the 80 pairs are not independent draws of prompts.

## 2026-10-15 queued work (nothing new started until arms11 ends; one process at a time)
- ms:arms11 (running, gpu 1): src/03_arm_compare.py for notes C..B, both arms, same config as the D# run above.
- ms:q2 (waits for arms11 to close, then runs sequentially): 04_validate_chroma.py (cpu; chroma-argmax vs labelled syntheory notes, 100 clips/note, writes chroma_validation.json) -> 05_cache_hidden.py --model small (gpu 1; caches mean-pooled hidden states for all 9936 clips, hidden_small.npz) -> 06_direction_methods.py --model small (cpu; meandiff / logistic / lda directions + per-layer val auc/test acc for all 12 notes, into directions_alt/ and results_alt/).
- 03_arm_compare_v2.py on the server = repo src/03_arm_compare.py with --method {rfm,meandiff,logistic,lda}; rfm behaviour and output names unchanged. it is a separate file so the running arms11 loop keeps using the original.
- caveat on 06: the split is rebuilt from 01's random call order (seed 42, random.sample on index lists) and is assumed, not verified, to select the same clips as the rfm runs. only split sizes are asserted (for D#).
- scripts only compile-checked on the mac (py_compile), not run.
