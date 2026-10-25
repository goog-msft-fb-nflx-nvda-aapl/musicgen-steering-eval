# WORKLOG: era / market stem analysis (topic 2)

Newest entries at the bottom. Every entry states config, data and the measured result; interpretation is labeled.

## 2026-10-25 setup
- Data: PA1 `dataset_A` (decade, 6 classes: 1960s to 2010s) and `dataset_B` (market, 6 classes: US, UK, Brazil, Spain, Germany, Italy; all 1980s). 30 s middle excerpts, WAV mono 24 kHz, artists disjoint across splits, test labels hidden. Labeled: A 1026 train / 132 val, B 798 train / 102 val. Source: Discogs-VI recordings with YouTube links (so audio is YouTube-quality).
- Prior result on the same data (PA1 README, validation): MERT-v2-30s frozen probe 0.5455 top-1 on A, 0.6471 on B (single-model numbers). Those are the reference points for the mix condition here.
- Server: `~/era_market/` with `venv` (system-site-packages of conda env musicgen_steer: torch 2.5.1, transformers 5.18; added demucs 4.1.0, pyloudnorm). Data zips copied by rsync from the Mac PA1 folder.
- Disk on server: 136 GB free of 1.8 TB (shared, 92% used). Stems and embeddings for 2292 clips are small (about 1 GB), keep it that way.

## 2026-10-25 pipeline smoke tests and launch
- Code: `src/01_separate.py` (HT Demucs `htdemucs`, shifts 1, overlap .25; clip 24k mono -> 44.1k stereo -> stems -> 24k mono fp16; accompaniment = drums + bass + other), `src/02_embed.py` (m-a-p/MERT-v2-30s, 24 layers x 1024, time-mean with feature_attention_mask; conditions mix_raw, mix_norm, voc_raw, voc_norm, acc_raw, acc_norm, recomb_norm; norm = -23 LUFS via pyloudnorm), `src/04_probe.py` (logistic regression on standardized layer features; layer and C chosen by 5-fold CV on train only, scored once on validation; bootstrap CI; exact McNemar between conditions), `src/run_stems.sh`.
- Smoke test (4 clips, GPU 2 shared with other users at ~100% utilization): separation about 1 s per clip including model load amortized (8.7 s wall for 4); MERT-v2-30s works under transformers 5.18 with output_hidden_states (no hook workaround), shape (24, 1024) per clip.
- Launched tmux `era` on the server: separate A then B, then embed A then B (all 7 conditions). Done flag: `~/era_market/stems_emb.done`. Estimated 40 min separation + embedding time.
- Added `src/03_stats.py` (9 signal statistics per clip for mix, vocal and accompaniment: LUFS, loudness range, crest factor, RMS, bandwidth, centroid, rolloff-85, zcr, noise floor), `src/05_langid.py` (openai/whisper-small language-token posterior over 100 languages on the vocal stem and the mix), `src/run_analysis.sh` (waits for `stems_emb.done`, then stats, language ID, probes). Language ID smoke test passed (3 clips).
- tmux on server: `era` (separate A then B, then embed), `era2` (waits, then analysis chain, writes `analysis.done`), `era3` (separate B in parallel on GPU 1; the script skips clips already separated). Measured about 20 clips/min per process on the shared GPUs, so separation takes roughly 1 to 1.5 hours.

## 2026-10-25 first results: stem-wise probes (MERT-v2-30s, validation split)
Config: `src/04_probe.py`. Features: MERT-v2-30s, time-mean per layer (24 x 1024). Probe: standardized logistic regression; layer and C chosen by 5-fold CV on the train split only (CV is not artist-grouped, so it only selects hyperparameters), then scored once on the validation split (A: n=132, 22 per class; B: n=102, 17 per class). 95% CI = bootstrap over clips (2000). McNemar = exact, paired on the same validation clips, uncorrected for multiple comparisons. Chance: 1/6 = 0.167. Separation: HT Demucs (vocals vs drums+bass+other), norm = -23 LUFS per signal. Clips come from YouTube so all audio is lossy and old recordings are transfers.

Validation accuracy (best layer in brackets):
| condition | A decade | B market |
|---|---|---|
| mix_raw | 0.515 [L17] (CI 0.43-0.60) | 0.578 [L14] (0.48-0.68) |
| mix_norm | 0.500 [L17] | 0.578 [L14] |
| vocals_raw | 0.394 [L20] | 0.549 [L15] |
| vocals_norm | 0.288 [L10] | 0.588 [L15] |
| accomp_raw | 0.508 [L10] | 0.412 [L13] |
| accomp_norm | 0.432 [L12] | 0.382 [L14] |
| recombined_norm | 0.508 [L17] | 0.569 [L14] |
Paired McNemar (validation): decade, mix_norm vs vocals_norm 41 vs 13 clips p=0.0002; accomp_raw vs vocals_norm 39 vs 10 p<0.0001; mix vs accomp not significant (p>=0.09). Market, mix_norm vs accomp_norm 25 vs 5 p=0.0003; vocals_norm vs accomp_norm 32 vs 11 p=0.002; mix vs vocals not significant. Recombined (vocals+accomp re-summed) is not distinguishable from the original mix in either task (accuracy 0.508 vs 0.515 and 0.569 vs 0.578), so separation processing itself does not change the probe result here.

Shortcut baselines (`src/03_stats.py`, `src/05_langid.py`, `src/06_extra_probes.py`; same protocol):
| input | A decade | B market |
|---|---|---|
| 9 signal statistics, mix (LUFS, loudness range, crest factor, RMS, bandwidth, centroid, rolloff, zcr, noise floor) | 0.311 | 0.255 |
| signal statistics, vocal stem / accomp stem | 0.227 / 0.303 | 0.225 / 0.225 |
| Whisper-small language posterior, vocal stem | 0.273 | **0.529** |
| Whisper-small language posterior, mix | 0.318 | **0.539** |
| accomp_norm embedding + vocal language posterior | 0.462 | 0.529 |
| mix_norm embedding + vocal language posterior | 0.485 | 0.578 |
Market: language posterior alone (0.53) is not significantly different from the full-mix embedding probe (0.578) by McNemar on 102 clips; adding the vocal-stem language posterior to the accomp embedding raises accuracy from 0.382 to 0.529 (19 vs 4 clips, p=0.003) and adding it to the mix embedding does not change accuracy (0.578). Decade: signal statistics and language posteriors reach 0.23-0.32, above chance but far below the embeddings (p<=0.002 vs mix_norm).

Per-class validation recall and confusions: `results/confusion_summary.txt`. Decade: errors are mostly adjacent decades (mix_norm within-one-decade accuracy 0.841, mean absolute decade error 0.75; 2000s is the weakest class at 0.27 recall, confused with 1990s and 2010s). Vocal stem decade error is larger (mean 1.23, within-one 0.705). Market: Brazil and Spain are easiest (mix 0.94, 0.65), Germany/US/Italy hardest; US->Germany and UK->US are the top mix confusions.

Reading (interpretation, one encoder, small validation sets, uncorrected pairwise tests):
- Decade information in this setup sits mostly in the accompaniment: accompaniment alone matches the mix, the vocal stem is clearly worse (but above chance). Market information sits mostly in the vocals and is largely explained by language: the vocal stem matches the mix, accompaniment is much worse, and a language-ID posterior alone almost reaches the mix accuracy.
- The accompaniment still carries some market information (0.38-0.41 vs chance 0.167) that is not language, since the accompaniment stem has no singing language except separation bleed; bleed of vocals into the accompaniment stem was not measured.
- Simple signal statistics do not explain the decade result (0.31 vs 0.50), but the loudness cue is partly removed already by the -23 LUFS normalization used in all norm conditions; raw and normalized conditions are not significantly different.
- Caveats: validation n is 102 to 132 so CIs span about +-9 points and differences below about 10 points are not resolved; 16+ pairwise tests were not corrected; label noise from reissues (Discogs release year) is not handled; PA1 test labels are hidden so there is no untouched test split here.

## 2026-10-25 second encoder and generation launch
- MuQ (OpenMuQ/MuQ-large-msd-iter, self-supervised on MSD, independent of Discogs-supervised MAEST) needed two workarounds under transformers 5.18: `_attn_implementation` on its EasyDict configs, and forward hooks on `m.model.conformer.layers` to rebuild hidden states (13 states x 1024). `src/muq_util.py`, `src/02b_embed_muq.py`. Embeddings for all 7 conditions of A and B running (tmux `era_muq`, flag `muq.done`). Purpose: (i) check the stem pattern is not MERT-specific, (ii) an era probe on an encoder different from the analysis encoder for evaluating generated covers.
- Generation: `src/07_generate.py`, MusicGen-Melody 1.5B fp16 (transformers MusicgenMelodyForConditionalGeneration), guidance 3.0, 1500 new tokens (30 s), sampling. Conditioning chroma = original MIX of a validation clip of dataset A; the generated audio replaces the accompaniment stem; remix = original vocal stem + generated audio RMS-matched to the original accompaniment. 12 clips (2 per decade, first in manifest order) x 4 prompts (neutral, 1960s, 1980s, 2010s) x 2 seeds = 96 generations; prompts in the script. Smoke test: 2 generations in about 1 minute after model download (tmux `era_gen`, flag `gen.done`, outputs `~/era_market/gen/`, local only: PA1 audio is licensed for the course, so no generated audio from it goes into git).

## 2026-10-25 vocal-bleed diagnostics (`src/08_bleed.py`, `src/09_bleed_analysis.py`, `results/bleed_summary.txt`)
Config: per clip vocal energy share = rms(vocal stem) / (rms(vocal) + rms(accomp)); Whisper-small language posterior computed on the loudness-normalized ACCOMPANIMENT stem, then the same logistic-regression protocol as the vocal-stem language probe (CV on train, scored once on validation).
- Language posterior of the accompaniment stem predicts market at 0.235 validation accuracy (cv 0.267; chance 0.167), versus 0.529 for the vocal stem. Reading (interpretation): little language information is left in the accompaniment stem, so the 0.38-0.41 market accuracy of accompaniment-only probes is probably not explained by bleed of sung language; it could still come from vocal timbre bleed or from musical cues. Not tested directly.
- Same posterior on decade: 0.326 (cv 0.307), which is about the level of the other weak shortcut baselines (0.27-0.32).
- Vocal share per clip: mean 0.35 (A) and 0.36 (B), full range 0 to 1 (some clips have almost no separated vocal). Validation accuracy by vocal-share tercile (n=34 to 44 per tercile, so differences below about 15 points are not resolved): market mix_norm 0.35 / 0.65 / 0.74, vocal 0.41 / 0.68 / 0.68, accompaniment 0.29 / 0.41 / 0.44 (low / mid / high); decade accompaniment 0.25 / 0.57 / 0.48. Market accuracy of the mix rises with vocal share (spearman 0.24 between vocal share and correctness), consistent with the language reading; the least-vocal tercile is also the least accurate for the accompaniment decade probe, which was not expected and is not explained (could be quiet or old recordings where separation finds little vocal).
