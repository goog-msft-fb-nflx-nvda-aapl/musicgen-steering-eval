# Metrics survey synthesis (topic 2): what we measure and how we validate it

Merges two replies to `CommE5070_FP_era_market_cover_metrics_survey_prompt.md` (raw: `survey2/topic2_survey2_claude.md`, `survey2/topic2_survey2_gemini.md`). Manual listening is only the last, qualitative round.

## 1. Reliability of the replies
| Reply | Verdict | Spot checks (2026-10-25, abstract pages only) |
|---|---|---|
| claude | Preferred: concrete, tables, exact metric definitions, a validation plan with acceptance criteria, power simulation. Several claims are its own design and say so. | Confirmed: Afchar et al. 2405.04181 (99.8% on real audio vs "fake reconstructions"); MuseCPEval 2512.14629 (ISMIR 2026; abstract names four facet categories and says objective validation plus a human study, the specific metrics named in the reply were not confirmed from the abstract); KAD 2502.15602 (arXiv preprint; "distribution-free, unbiased", faster convergence at small n; the reply's "ICML 2025" was not confirmed); COCOLA 2404.16969 (evaluates accompaniment generation; checkpoints on MUSDB18-HQ, MoisesDB, Slakh2100, CocoChorales). Not confirmed: 2609.04516 is a real pitch-class steering paper, but its abstract does not mention rotation controls, so "rotation chance level is established" stays unverified (the idea is still sound and cheap). |
| gemini | Useful ideas (selectivity control tasks, Demucs re-extraction SI-SDR for masking, LMM random intercept, CoverHunter) but weaker sourcing: it echoes our own numbers (chroma DTW 0.78 to 0.88) as if from the literature, says codec artifacts make classifiers call everything "post-2000s electronic" (ArtifactNet 2604.16254 exists but its abstract is about detection, not era), proposes a "200 to 300 Hz low-pass" for codec artifacts, which makes no sense, and states a 24-item power requirement without a checkable derivation. Treat as brainstorm. |

## 2. Diagnosis of our three problems (agreed by both replies, mechanism partly unverified)
1. Era probe reads "2000s" on all generated audio: expected under codec/decoder artefacts (Afchar et al. show decoder reconstructions are trivially detectable). Fix = put the real references through the same EnCodec path and score relative to the neutral prompt.
2. MERT vs MuQ disagreement and MuQ's response to the DSP filter: report as a finding, keep the DSP filter as a standing counterfactual control.
3. Chroma floor: do not report raw chroma/DTW similarity. Report an identification score (rank of the song's own original among all originals) and a lift over a shuffled-song floor.

## 3. Metric suite (decided)
| Construct | Primary | Secondary | Notes |
|---|---|---|---|
| Era presence | relative, codec-matched probe score: Delta_X = P(X | prompt X) - P(X | neutral), probes trained on EnCodec-resynthesized real accompaniment, MERT and MuQ separately | KAD own-era margin with codec-matched references (kadtk); Discogs-style tagger (MAEST/Discogs-EffNet) mapped to decades on real data; production features as shortcut detectors only | era-presence validity must pass the controls in section 4 first |
| Content preservation | song identification: top-1 and MRR of each generated accompaniment among the 12 originals (chroma-DTW, high-passed at middle C), lift over shuffled floor | chord agreement (mir_eval majmin, rotation floor), folded delta-BPM, CSI embedding rank (CLEWS/CoverHunter) | the vocal is kept, so vocal F0 accuracy is trivially high; measure the accompaniment lead line instead |
| Quality | Audiobox Aesthetics PQ and CE (per item, no reference) | KAD vs codec-matched real accompaniment | FAD is biased at n = 12 to 36, use KAD |
| Remix fit | COCOLA(vocal, new accompaniment) with ceiling (original pair) and floor (cross-song pair) | vocal-to-accompaniment loudness ratio (pyloudnorm), SI-SDR of the vocal re-extracted by Demucs from the remix, key match | no accepted standard; label as proxies |
| Analysis attribution | balanced accuracy, MAE-decade, QWK, log-loss per input; exact McNemar mix vs stem with Holm | two-stem Shapley; MDL online code length; control-task selectivity | artist ids are anonymized in PA1, so artist-cluster bootstrap and per-artist control tasks are not possible: use clip-level bootstrap and the original Hewitt and Liang random-label control task |

## 4. Validation of the metrics before use (no listeners)
Positive controls (real era-X audio through EnCodec; real era-X accompaniment swapped into the remix), negative controls (neutral prompt, shuffled era labels of references, pink noise), DSP counterfactual (reported as a ratio to generation), sensitivity curve (crossfade original to generated, 5 steps, expect monotone change), seed ICC (average more seeds if ICC < 0.5), encoder agreement in sign. Acceptance (proposed, to fix before running): positive beats negative with a bootstrap CI excluding 0; decade-ordering Spearman >= 0.5 on the codec positive control; monotone sensitivity (Spearman >= 0.8); both encoders agree in sign of Delta per era, otherwise report the metric as encoder-dependent; content metric at chance on the oracle-swap and near 1 on the DSP control.

## 5. Statistics
Unit = song, seeds averaged inside (song, condition), exact paired sign-flip or Wilcoxon across songs, mixed model with a song random intercept as robustness, Holm over the 3 eras, one pre-declared primary metric per construct. Power (claude reply, own simulation, normal differences): 12 songs detect d about 1.0 with 0.88 power at alpha 0.05 and 0.72 after a 3-way correction; 20 songs give 0.97 at d = 1.0 after correction. Smallest attainable exact two-sided p with 12 songs is about 0.0005. Adding songs helps far more than adding seeds. Decision: go from 12 to 24 songs (generation took about 1 to 2 minutes per clip on the shared GPU).

## 6. Implementation order (about 6 person-days in the reply)
1. EnCodec resynthesis of real train and validation accompaniment (MusicGen 32 kHz compression model), re-embed with MERT and MuQ, retrain era probes, rerun generation evaluation with Delta scores (GPU).
2. Controls a to f and acceptance check.
3. Song-identification and chord/tempo metrics on the existing 96 generations (CPU), then Audiobox Aesthetics, COCOLA, KAD (GPU).
4. Analysis-half metrics (QWK, MAE, log-loss, MDL, selectivity) on stored embeddings (CPU).
5. Scale to 24 songs, rerun, then final qualitative listening round: audit top and bottom 2 items per metric, not random items.

## 7. Unverified or uncertain
Items the replies flag (EnCodec resynthesis API for the MusicGen compression model, chord recognizer choice and licence, COCOLA behaviour on vocal-accompaniment pairs, CLAP era alignment which neither reply could find validated, style-to-decade mapping quality) plus: KAD venue, MuseCPEval metric list, rotation-control claim, Gemini's numeric power and artefact claims. Acceptance thresholds are proposals, not literature standards.
