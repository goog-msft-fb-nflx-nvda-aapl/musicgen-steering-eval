backup of server-side artifacts that would be lost when gsm-gpu2 access ends (copied 2026-10-17 from ~/musicgen_steer):
- trained_concepts/musicgen_small/{directions,results}: rfm note directions + detectors/metrics for all 12 notes (01_train_note_direction.py)
- directions_alt/, results_alt/: meandiff/logistic/lda directions and per-layer metrics (06_direction_methods.py)
- mert_probe.json/.pkl: MERT-v1-95M note probe (10a_mert_probe.py)
- logs/: all server *.log files at copy time
not backed up (large, regenerable): hidden_small.npz (448 MB, 05_cache_hidden.py), mert_feats.npz (350 MB), demo flac clips.
