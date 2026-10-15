# todo

- [x] gsm-gpu2: gpu check, smoke-test gpu 2, dir, env
- [x] music-rfm cloned + installed
- [x] note D# direction on musicgen-small trained (see WORKLOG 2026-10-14)
- [x] xRFM 'no split/0 iterations' lines are the cross-layer aggregation stage (100 hyperparam fits), per-layer probes are not logged; generation hooks checked, no 48-layer assumption but library hook breaks on transformers 5 (own hook used)
- [x] first pilot generation D# (WORKLOG 2026-10-14)
- [ ] listen to eta>=1.0 clips / add quality metric (clap, mad) to see why hits drop to 0
- [ ] validate chroma evaluator on held-out syntheory note clips (real label check)
- [ ] train directions for other notes (all 12) for the real pilot grid; finer eta grid between 0.1 and 0.8
- [ ] decide how server gets private repo (deploy key vs scp)
- [ ] read closest prior steering papers (verify novelty list)
- [ ] decide independent evaluator per concept (see PLAN open items)
- [ ] pilot: note direction, small, 20x12, eta 0/0.3/0.6, throughput small+large
- [ ] pitch deck inputs from PLAN.md

- [ ] after train12 finishes: faithful musicrfm config arm (exp weighting kappa .95, p=0.3, eta .15/.3/.45/.6) on D# vs my constant top-8 config, paired
- [ ] read 2609.22135, 2604.15557, 2504.04479 properly
- [ ] direction-method interface: mean-diff, logistic, lda next to rfm (needs hidden-state cache; plan one extraction pass)
- [ ] train12 (C..B except D#) running on gsm-gpu2 tmux ms:train12, logs note_small_*.log

- [x] all-notes rfm arm sweep, chroma validation, alt direction fits (WORKLOG 2026-10-15)
- [ ] altarms running on gpu 2: logistic/lda/meandiff steering, then analysis_arms-style paired comparison across methods
- [ ] investigate B (const8 flat at ~0.22) and D# (weakest note)
- [ ] fidelity metric (clap/mad) before any matched-quality comparison; evaluator is only 86% on clean notes (fifth bias)
- [ ] medium/large confirmation subset (large throughput not measured)
