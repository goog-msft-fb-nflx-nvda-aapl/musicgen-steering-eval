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

## 2026-10-15 all-notes arm sweep, chroma validation, alternative direction fits (all musicgen-small)
runs finished 2026-10-15 ~09:05 server time (arms11 loop, then q2 chain: 04 -> 05 -> 06). arms sweep: 03_arm_compare.py (orig version on server; same config as the D# entry above) for the 11 other notes, gpu 1; 05 cache on gpu 1; 04/06 cpu. files: results/arms/musicgen_small_<note>_arms.json, results/chroma_validation.json, results/alt_directions/<method>/small/note_<note>.json, analysis: src/analysis_arms.py (reads the json, cpu).

### chroma evaluator validation (04_validate_chroma.py)
syntheory 'notes' clips, 100 per note (1200 total, seed 0), resampled to 32 kHz, argmax of time-mean librosa chroma_cqt vs labelled root note. overall accuracy 0.8625 (Wilson 95% CI 0.842-0.881); per note 0.80 (F#, G) to 0.94 (C#); C 0.93, C# 0.94, D 0.88, D# 0.88, E 0.92, F 0.82, F# 0.80, G 0.80, G# 0.81, A 0.84, A# 0.83, B 0.90. the commonest errors are perfect-fifth confusions: true F# read as C# (18), G as D (16), F as C (13), A as E (13), G# as D# (11), A# as F (9), B as F# (7). a librosa warning "Trying to estimate tuning from empty frequency set" appeared. so on clean single-note clips the evaluator is wrong about 14% of the time, with a systematic bias to the fifth above the root; this applies to every "hit" number in this log. it was validated on synthetic labelled notes only, not on musicgen output or real polyphonic music.

### all 12 notes, rfm directions, const8 vs musicrfm-style arm (80 paired clips per note x arm x eta; hit = target note is chroma argmax)
unsteered hit rate per note: C .138, C# .037, D .212, D# .013, E .088, F .113, F# .037, G .150, G# .050, A .088, A# .025, B .050 (pooled 0.083).
hit rate at eta .15/.3/.45/.6:
- const8: C .57/.70/.81/.54; C# .49/.80/1.00/1.00; D .53/.78/.93/.95; D# .07/.29/.17/.11; E .54/.74/.84/.97; F .15/.51/.88/.96; F# .54/.76/.80/.41; G .40/.65/.80/.85; G# .44/.60/.78/.62; A .50/.79/.99/1.00; A# .49/.81/.96/.93; B .25/.21/.23/.21
- musicrfm-style: C .50/.75/.91/.95; C# .35/.53/.65/.81; D .24/.54/.75/.72; D# .04/.19/.26/.47; E .25/.50/.66/.81; F .14/.33/.62/.82; F# .10/.31/.49/.59; G .41/.66/.81/.91; G# .26/.50/.71/.80; A .53/.71/.86/.94; A# .38/.64/.84/.86; B .17/.46/.68/.68
pooled over the 12 notes (960 pairs per cell), exact McNemar vs unsteered, Holm over 8 cells: const8 hit .414/.636/.765/.714, musicrfm-style .280/.509/.688/.781 (unsteered .083); every cell holm p < 2e-31 (discordant up vs down, e.g. musicrfm-style eta .6: 683 up, 13 down).
arm vs arm pooled (musicrfm-style minus const8), Holm over 4 etas: const8 higher at .15 (.414 vs .280, p=2e-10) , .3 (.636 vs .509, p=4e-9), .45 (.765 vs .688, p=8e-5); musicrfm-style higher at .6 (.781 vs .714, p=1.8e-4).
observations: D# (the note used in the earlier pilot and D# arm entry) is the weakest note for both arms; most other notes reach 0.8-1.0 at eta .45-.6. const8 on B stays at 0.21-0.25 at every eta (no trend with eta); const8 drops at eta .6 for C (.81->.54), F# (.80->.41) and G# (.78->.62), while the musicrfm-style arm rises monotonically with eta for 10 of 12 notes (D, B plateau/dip at .6). caveats: one model size, evaluator error ~14% with fifth bias (hits for a target note can also arise from clips whose true root is a fourth below it; not quantified), pairs are not independent draws of prompts (20 prompts x 4 seeds per note), no fidelity metric yet, so "hit rate" says nothing about whether the audio stayed musical. the two arms differ in total injected magnitude, layer weights and injection probability together (interpretation only: not separated).

### alternative direction fits (06_direction_methods.py, cached hidden states, same task + split sizes as rfm; split identity not clip-verified)
test accuracy (%) of the 1-d projection at the layer with best val AUC, mean over 12 notes (min-max): meandiff 87.0 (83.2-90.0), logistic 92.3 (89.2-95.6), lda 92.9 (90.0-95.2); for reference the rfm top direction at its best-val-score layer: 92.5 (90.4-96.0). n=250 test clips per note, SE ~2 points. different layers are selected per method/note. this is decodability only; whether the directions steer is not measured yet.

### running
ms:altarms on gsm-gpu2 gpu 2 (idle when checked; passed matmul check; other users on gpus 0,1,3): 03_arm_compare_v2.py for methods logistic, lda, meandiff x 12 notes, same arms/etas/seeds. ~5 min per note -> ~3 h. outputs ~/musicgen_steer/arms/musicgen_small_<note>_<method>_arms.json.

## 2026-10-15 direction methods x arms, pooled over 12 notes (src/analysis_methods.py; runs: ms:altarms, 03_arm_compare_v2.py --method <m>, gpu 2, 36 runs, no tracebacks)
config as the rfm arm sweep above (musicgen-small, per-note direction from 06_direction_methods.py for logistic/lda/meandiff; unit-norm direction, same layer weighting rule using each method's own val AUC; eta in .15/.3/.45/.6; 20 prompts x 4 seeds x 12 notes = 960 pairs per cell; unsteered baseline identical across method runs, hit 0.083). hit = target note is the chroma argmax.
pooled hit rate at eta .15/.3/.45/.6:
- rfm: const8 .414/.636/.765/.714; musicrfm-style .280/.509/.688/.781
- logistic: const8 .311/.509/.607/.594; musicrfm-style .180/.343/.536/.678
- lda: const8 .225/.356/.435/.433; musicrfm-style .131/.229/.328/.440
- meandiff: const8 .773/.939/.974/.932; musicrfm-style .692/.908/.971/.985
all 32 cells differ from the unsteered baseline (max Holm p 3.4e-4). vs the rfm direction at the same arm/eta (exact McNemar, Holm over 24 cells): meandiff higher in all 8 cells (e.g. musicrfm-style eta .6 .985 vs .781, 206 method-only vs 10 rfm-only hits), logistic and lda lower in all 16 cells (all holm < 1e-6).
target share of chroma energy (frac) minus unsteered, eta .6: rfm +.125 (const8) / +.150 (musicrfm-style); logistic +.096/+.109; lda +.063/+.067; meandiff +.360/+.295.
per-note, eta .6, musicrfm-style: meandiff .95-1.00 for every note; rfm .47 (D#) to .95 (C); lda .21 (F#) to .61 (A). const8 meandiff at A# only .51 (other notes .85-1.00).
decodability (earlier entry) did not order the methods the same way as steering: mean-diff had the lowest probe test accuracy (87.0% mean) yet the highest hit rate; lda/logistic/rfm decoded at ~92-93% and steered less. observation only; one model size, one evaluator.
what this does NOT show: same nominal eta is not the same perturbation size or the same audio quality. the directions are unit norm but residual-stream geometry differs by method; the large chroma-share gain for meandiff (+.30-.36 of the chroma energy) could be steering or could be degraded audio (e.g. a drone on the target pitch). no fidelity metric has been run on these clips, so the matched-quality comparison (plan Q2) is still open. the chroma evaluator is 86% accurate on clean notes (fifth bias, see above). also: a new VLLM::EngineCore process owned by jtan (pid 796576, 126 GB) appeared on gpu 0 after i stopped the earlier one; not launched by me, left alone.
next: add clap-based fidelity (text-audio score + embedding distribution distance vs unsteered) and re-generate the same seeds to get it.

## 2026-10-15 fidelity sweep launched (src/07_fidelity.py)
regenerates the same clips as the arm runs with seeds 0 and 1 only (half of the 4 seeds; 20 prompts x 2 seeds x 12 notes = 480 pairs per cell) for all 4 methods x 2 arms x 4 etas, and stores per clip a CLAP (laion/clap-htsat-fused, HF transformers) audio embedding + text-audio cosine with the generation prompt, plus chroma hit/frac. the laion/larger_clap_music checkpoint could not be loaded (pickle .bin refused on torch 2.5.1); htsat-fused is the same family as musicrfm's 630k-audioset-fusion-best but not necessarily the identical checkpoint. smoke test D#/meandiff/seed 0: regeneration reproduced all 180 earlier hit labels (repro_hit_match 1.0), so seeds + hooks are deterministic. window ms:fid on gpu 2 (free when checked), order rfm, meandiff, logistic, lda x 12 notes, ~3 min per run, ~2.5 h. outputs ~/musicgen_steer/arms_fid/*.npz. the smoke-test output was deleted by my own cleanup before i inspected the clap numbers.

## 2026-10-15 fidelity results (src/analysis_fidelity.py -> results/fidelity/)
config: musicgen-small, gen seeds 0,1 x 20 prompts x 12 notes = 480 pairs per cell, same directions/arms/etas as the arm sweeps; regeneration reproduced all earlier hit labels in all 48 runs (repro_hit_match 1.0, n=360 each). fidelity = laion/clap-htsat-fused text-audio cosine with the generation prompt, and clap-embedding distances to the unsteered clip of the same (note, seed, prompt). MMD^2 = unbiased rbf, median bandwidth, steered vs unsteered embeddings (n=480 each). files: results/fidelity/fidelity_summary.json, pareto.png (hit vs each fidelity metric, eta .15-.6 along each line), npz/ (raw embeddings + per-clip rows).
pooled hit / dCLAP (steered minus unsteered text-audio score, paired wilcoxon) / MMD^2, selected cells:
- rfm const8 eta .45: .752 / -.066 / .077; rfm musicrfm-style eta .6: .821 / -.013 / .028
- meandiff const8 eta .15: .785 / -.015 / .014; meandiff musicrfm-style eta .15/.3/.45/.6: .777/.963/.983/.985, dCLAP -.005/-.025/-.042/-.060, MMD^2 .014/.040/.064/.095
- logistic musicrfm-style eta .6: .723 / -.009 / .018; lda musicrfm-style eta .6: .458 / -.003 / .015
observations (measured, one model, one clap checkpoint, chroma evaluator): at roughly equal MMD^2 or dCLAP meandiff reaches higher hit rate than rfm (e.g. MMD^2 ~.014: meandiff .78 vs rfm ~.4-.6 on the same arm curve; meandiff musicrfm-style eta .3 dCLAP -.025 hit .963 vs rfm const8 eta .3 dCLAP -.023 hit .637). so the meandiff advantage is not explained only by a larger clap-measured departure. meandiff still loses clap text-audio score at high eta (const8 eta .6: -.190, hit falls .977 -> .925). rfm/logistic/lda musicrfm-style arms keep dCLAP within ~.01 up to eta .6 but also hit less. const8 is costlier in fidelity than musicrfm-style at the same eta for every method.
caveats: paired twin cosine distance has a noise floor (~.24-.29 even at eta .15 for all methods, sampling chaos), so MMD^2 and dCLAP are the more informative axes; clap measures prompt agreement and embedding shift, not note quality, and a drone on the target pitch is not excluded (no listening yet; demo flacs on server arms_fid/*.flac, not pulled). perturbation norm is not matched across methods: that control is next (todo).

## 2026-10-16 controls (src/08_controls.py, results/fidelity/npz/*_random_ctl.npz, *_shuffled_ctl.npz)
config: same as the fidelity sweep (musicgen-small, 12 notes x gen seeds 0,1 x 20 prompts = 480 pairs/cell, arms const8/musicrfm-style, eta .15-.6, same hooks and clap/chroma features). layers and weights from the meandiff val scores of each note. random = unit gaussian direction per layer (seeded by note+layer). shuffled = meandiff on the same balanced train split with class labels permuted. all directions are unit norm, so the nominal injected norm equals that of the real methods at equal eta/weight. both 12-note loops ran without errors (24 npz).
pooled hit (baseline .083) / dCLAP / MMD^2:
- random: hit .067-.115 in all 8 cells (e.g. musicrfm-style eta .6 .115, dCLAP +.003; const8 eta .6 .102, dCLAP -.028). no hit increase beyond the baseline range seen here.
- shuffled-label meandiff: const8 .167/.177/.206/.250 (eta .15..6), dCLAP -.070/-.180/-.229/-.231, MMD^2 .043/.110/.156/.200; musicrfm-style .140/.171/.163/.135, dCLAP -.041/-.093/-.147/-.178.
- for comparison real meandiff const8 eta .45: hit .977, dCLAP -.126, MMD^2 .126; musicrfm-style eta .6: .985, -.060, .095. rfm const8 eta .45: .752, -.066, .077.
reading (interpretation, one model, one evaluator): a random unit direction at these etas leaves hit near baseline with little clap change, so the hit gains of the real directions need note-specific content, not just any perturbation of that norm. a label-free dominant-data direction (shuffled meandiff) costs the most clap similarity (-.23 at eta .45-.6) and raises hit only to .14-.25, so large clap loss does not by itself produce chroma hits; meandiff with real labels reaches .98 at roughly half of that clap loss. residual risk: a drone-like target-pitch collapse is not excluded (no listening); hit is chroma argmax of the whole clip.

## 2026-10-17 Q3 layer sweep, meandiff directions (src/09_layer_sweep.py, src/analysis_layers.py, results/layer_sweep/*_meandiff_sweep.json)
config: musicgen-small, meandiff note-vs-rest directions (unit norm), 12 notes, one note per run. stage 1 (dev prompts 0-9, each twice in one batch of 20 clips, gen seed 0): steer ONE layer at a time (weight 1, p=1, eta .6 and 1.5), all 23 layers (-1..-23); causal score = mean hit over the two etas minus the unsteered dev hit. stage 2 (held-out prompts 10-19, twice each): 8-layer sets at weight 1, p=1, eta .15/.3/.45: probe8 = top-8 by meandiff val AUC (same rule as const8 elsewhere), causal8 = top-8 by stage-1 causal score, rand8 = seeded random 8. layer choice never sees held-out prompts. hit = chroma argmax = target. only 20 clips per cell per note (pooled 240), so per-note numbers are noisy.
results:
- spearman(probe val score, causal score) across the 23 layers, per note: mean -0.01, range -0.55 (G#) to +0.45 (F#); 5 of 12 notes negative. mean probe8/causal8 overlap 3.2 of 8 layers.
- layer-averaged (12 notes): probe val AUC is highest mid-depth (-12..-9, ~.93-.94) and flat-to-lower at the end; single-layer causal gain is highest at the last layers (-1 +.89, -2 +.88, -5 +.79, -6 +.74), lowest around -15..-9 (~+.29-.34), and rises again at the first layers (-23 +.54).
- held-out hit (pooled 12 notes, unsteered .083) probe8/causal8/rand8: eta .15 .763/.954/.771 (wilcoxon on 12 per-note values: causal8>probe8 p<.001, causal8>rand8 p=.006, probe8 vs rand8 p=.88); eta .3 .942/.983/.958 (all p>.1); eta .45 .987/.996/1.000 (all p>.18).
reading (interpretation): in this setup the layers a probe finds most decodable are not the layers where steering has the most effect, and a probe-selected 8-layer set did no better than 8 random layers at the lowest eta. causally chosen layers were mostly late layers, so the gain could just be "late layers are better steering sites"; a fixed last8/first8 control is being run (stage2_only, etas .075/.15/.3, ms:sets) to check that. only meandiff directions so far; rfm and the other methods not run. eta>=.3 cells are near 1.0 hit for every set and cannot discriminate sets.

## 2026-10-17 Q3 follow-ups: fixed-layer control (meandiff) and rfm sweep (results/layer_sweep/*_sets.json, *_rfm_sweep.json)
config as the meandiff sweep above (09_layer_sweep.py, 12 notes, dev prompts 0-9 for layer choice, held-out prompts 10-19 x2 for evaluation, 20 clips per cell per note, weight 1 p=1). *_sets.json: probe8/causal8/rand8 plus fixed last8 (layers -8..-1) and first8 (-23..-16), etas .075/.15/.3. rfm sweep: same stage 1 with the rfm direction (top eigendirection), stage 2 etas .075/.15/.3.
meandiff held-out hit pooled over notes (probe8/causal8/rand8/last8/first8): eta .075 .450/.592/.417/.612/.250; eta .15 .763/.954/.771/.971/.525; eta .3 .942/.983/.958/.996/.783. wilcoxon over 12 per-note values: last8 > probe8 p=.007 (.075), <.001 (.15), .072 (.3); causal8 vs last8 p=.10/.50/.18 (no detectable difference); probe8 vs rand8 p=.28/.88/.55.
rfm (stage-1 sweep, held-out hit probe8/causal8/rand8): eta .075 .242/.400/.283; .15 .442/.708/.462; .3 .588/.929/.662; causal8 > probe8 p=.021/<.001/.001, probe8 vs rand8 p>.37 in all. spearman probe vs causal per note mean +0.07 (range -0.30 to +0.55), mean overlap 2.4 of 8.
reading (interpretation): the causal sweep's advantage over probe selection is matched by simply taking the last 8 layers (meandiff), so what the data shows is "late layers steer better than the layers with the best probe score", not that a per-note causal sweep adds value beyond a depth rule. a last8 control for rfm is running (ms:sets_rfm, gpu 1). per-note causal rankings come from 20 dev clips and are noisy, which limits what a per-note sweep could show. still no predictor of steering-effective layers other than depth.
