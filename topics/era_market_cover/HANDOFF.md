# Handoff for a session running on the GPU server (topic 2: era / market)

Read first: `TODO.md`, `WORKLOG.md`, `SURVEY.md` in this folder. Repo clone on the server: `~/final_project` (push with `~/bin/gpush origin main`; token file `~/.config/secrets/github_token`, mode 600, never print or commit it). Hugging Face token: `~/.cache/huggingface/token` (mode 600; account jameswatanabegoogle2024; upload derived features only, never course audio or stems).

Working dir for experiments: `~/era_market` (venv `~/era_market/venv`, data `data/ex`, stems `stems/`, embeddings `emb/`, generated audio `gen/`, logs `logs/`). Scripts live in `~/final_project/topics/era_market_cover/src/` and are copied to `~/era_market/src/` (keep the two in sync, repo is the source of truth). Use GPU 1 or 2 only, one or two jobs at a time, do not touch other users' files or processes.

State at handoff (2026-10-25): stem-wise MERT-v2 probes, shortcut baselines, language ID and bleed analyses are done and logged. Running or just finished: MuQ embeddings (`emb/*_muq_*.npy`, flag `muq.done`) and 96 MusicGen-Melody generations (`gen/`, flag `gen.done`).

Next steps: (1) run `04_probe.py` on the MuQ embeddings (extend it to read the `_muq_` file names and 13 layers) and compare the stem pattern with MERT-v2; (2) evaluate the generated audio: melody preservation (chroma DTW between original accompaniment and generated, pitch-track accuracy on the vocal stem is unchanged by construction), era probe on generated accompaniment using a probe trained on the other encoder, DSP-only "old/new" baseline, paired tests; (3) upload embeddings and stats to a private HF dataset; (4) log everything in `WORKLOG.md`, tick `TODO.md`, commit and push.

Rules (from the project CLAUDE.md): report only measured numbers; label interpretation as interpretation; no Claude or Anthropic attribution in commits or repo files; keep chat terse and put substance in the worklog.
