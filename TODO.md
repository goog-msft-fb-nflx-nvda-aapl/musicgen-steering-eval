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
