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

## queue (decided 2026-10-15, do in order; gpu server ends oct 31)
- [x] fid sweep done, analysed (WORKLOG 2026-10-15 fidelity results)
- [x] controls done (WORKLOG 2026-10-16); direction geometry (cosine between methods, variance subspace of logistic/lda) still to do, cpu-only on cached hidden states
- [ ] Q3: per-layer causal steering sweep (dev prompts, eval held-out) vs probe-acc layers vs cheap gradient/attribution predictor of steering effect
- [ ] other syntheory concepts (chords, scales/modes, tempo) with mean-diff + rfm; then two-concept composition (additive vs interfering)
- [ ] second evaluator (mert/muq probe on real labeled audio or essentia), check method ranking holds
- [ ] medium confirmation of best configs (measure throughput first)
- [ ] write-up inputs from worklog; keep scripts+results pushed; clean server /tmp after each cycle
- [x] Q3 meandiff+rfm layer sweeps and last8/first8 controls done (WORKLOG 2026-10-17); open: gradient/attribution predictor of steering-effective layers; per-note sweeps are noisy (20 dev clips)
- [ ] second evaluator: 10a_mert_probe.py running on gpu 2 (ms:mert; note: tmux env lacked HF_HOME so it downloaded syntheory/mert to ~/.cache/huggingface, delete after); then run 10b_second_eval.py (written, untested) for 4 methods x 12 notes on gpus 1+2, compare hit rankings chroma vs mert
- [ ] second evaluator: mert probe done (test acc 90.6% on syntheory notes, layer 9; WORKLOG pending); 10b running 2026-10-17 on gpu 1 (rfm, meandiff) and gpu 2 (logistic, lda), ~1.6 h; then pull arms_eval2/*.npz -> results/eval2/, run src/analysis_eval2.py, log
