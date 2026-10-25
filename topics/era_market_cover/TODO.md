# TODO: era / market stem analysis (topic 2)

Context: `SURVEY.md` (design), `WORKLOG.md` (what ran, with configs and results). Data: PA1 Discogs-VI clips (A: 6 decades, 1026 train + 132 val labeled; B: 6 markets in the 1980s, 798 train + 102 val labeled; test labels hidden; splits are artist-disjoint; 30 s mono 24 kHz). Server dir `~/era_market` (venv `~/era_market/venv`, data `~/era_market/data/ex`). GPU access ends 2026-10-31; after that analysis runs on cached embeddings on the Mac.

## Now (2026-10-25: MuQ probes done, generation 107/96+ files present, evaluation of generated audio next)
- [x] unzip data on server, smoke test HTDemucs (2292 clips) and MERT-v2-30s under transformers 5
- [x] 01 separate all clips into vocals / accompaniment (24 kHz mono, fp16 npz), log runtime per clip
- [x] 02 embed 7 conditions with MERT-v2-30s (mix raw/norm, vocals raw/norm, accompaniment raw/norm, recombined)
- [x] 03 signal-statistics probe (loudness, crest factor, rolloff, bandwidth) as shortcut baseline
- [x] 04 probes (logreg, layer + C chosen by CV on train, scored once on validation), bootstrap CI, paired McNemar between conditions
- [x] 05 Whisper language ID on vocals and mix, language-only baseline for market, nested comparison with embeddings

## Next (results so far in WORKLOG 2026-10-25: decade lives in accompaniment, market in vocals/language)
- [ ] artist-grouped confidence: bootstrap over validation with multiple-comparison correction; second seed of probe selection
- [x] vocal-bleed check (WORKLOG 2026-10-25); instrumental-only subset still too small to test (tercile n=34-44)
- [ ] second encoder (MAEST, MuQ) to see if the stem pattern is encoder specific
- [ ] layer-wise curves per condition (data in probe json `layer_cv_best_by_layer`) as a figure
- [ ] second encoder family for evaluation of generated covers (MAEST or MuQ) so the era probe is not the analysis encoder
- [x] (first pass, 12 songs, WORKLOG 2026-10-25) MusicGen-Melody accompaniment regeneration: 5 to 12 songs x 3 target eras, keep original vocal stem, remix
- [x] (chroma only) melody preservation metrics (chroma DTW, RPA on vocal pitch track), era probe on generated audio, DSP-only "old" baseline
- [ ] decide on Discogs-VI-YT clique experiment (needs audio download by a teammate)

## Later / Mac only
- [ ] confusion analysis by decade pair and market pair, figures, report
- [ ] listening test (2AFC, about 12 listeners) if time

## Constraints
- keep data under `~/era_market` (jtan space), do not touch other users, one or two jobs at a time, GPU 1 or 2 only
- PA1 audio is licensed for the course: do not commit audio or embeddings; commit scripts and result tables only

## Open after first generation pass
- [ ] listening test / at least informal listening of gen/ remixes (audio is on the server, not in git)
- [ ] DSP baseline sensitivity: why MuQ moves with filters and MERT does not
- [ ] retrain the era probe with generated-style augmentation or compare to neutral-prompt anchor only
- [ ] more songs and prompt variants (2000s/1990s/1970s), guidance sweep; Discogs-VI cliques as real-reference targets
- [ ] upload embeddings/stats to private HF dataset (token on Mac and server)

## Metrics plan (SURVEY_metrics.md, 2026-10-25) in order
- [x] EnCodec-resynthesize real accompaniment, re-embed (MERT, MuQ), retrain era probes, rescore generations (WORKLOG 2026-10-25 codec); per-song Delta statistic and ICC still to compute
- [x] (12-song pass, WORKLOG 2026-10-25 validation) controls a-f; rerun on 24 songs with 3 seeds after generation finishes
- [x] song-identification, chord, tempo metrics on existing 96 generations (CPU) (WORKLOG 2026-10-25)
- [ ] Audiobox Aesthetics, COCOLA, KAD (kadtk) on generations
- [ ] analysis-half metrics: QWK, MAE-decade, log-loss, MDL, control-task selectivity on stored embeddings
- [x] (24 songs x 2 seeds done 2026-10-26; seed 2 running) scale generation to 24 songs; final qualitative listening round only after the quantitative suite is frozen

## From course materials (COURSE_MATERIALS_NOTES.md)
- [ ] guidance-scale sweep (1, 3, 5, 7) for MusicGen-Melody: style strength vs content preservation
- [ ] parametric production-only control (EQ + compressor + reverb) replacing the crude DSP filter; check Fx-Encoder++ / Diff-MST availability as era production metric and non-generative baseline
- [ ] MuseControlLite (Stable Audio Open) as second generator
- [ ] BS-RoFormer or UVR as second separator; lead vs backing vocals for the market result
- [ ] basic-pitch / CREPE melody metrics on the accompaniment lead line
- [ ] sparse autoencoder on stem embeddings (course idea 5, explainability), optional
- [ ] demo clip set for the oral presentation (audio required even for analysis); keep audio out of the public repo
- [ ] learnable layer-weighted-sum probe (MERT paper protocol) as robustness check of per-layer CV choice
- [ ] HPSS split of accompaniment stem (harmonic vs percussive), probe both parts for decade (cheap, CPU)
