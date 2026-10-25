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
