# TODO: era / market stem analysis (topic 2)

Context: `SURVEY.md` (design), `WORKLOG.md` (what ran, with configs and results). Data: PA1 Discogs-VI clips (A: 6 decades, 1026 train + 132 val labeled; B: 6 markets in the 1980s, 798 train + 102 val labeled; test labels hidden; splits are artist-disjoint; 30 s mono 24 kHz). Server dir `~/era_market` (venv `~/era_market/venv`, data `~/era_market/data/ex`). GPU access ends 2026-10-31; after that analysis runs on cached embeddings on the Mac.

## Now
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
- [ ] MusicGen-Melody accompaniment regeneration: 5 to 12 songs x 3 target eras, keep original vocal stem, remix
- [ ] melody preservation metrics (chroma DTW, RPA on vocal pitch track), era probe on generated audio, DSP-only "old" baseline
- [ ] decide on Discogs-VI-YT clique experiment (needs audio download by a teammate)

## Later / Mac only
- [ ] confusion analysis by decade pair and market pair, figures, report
- [ ] listening test (2AFC, about 12 listeners) if time

## Constraints
- keep data under `~/era_market` (jtan space), do not touch other users, one or two jobs at a time, GPU 1 or 2 only
- PA1 audio is licensed for the course: do not commit audio or embeddings; commit scripts and result tables only
