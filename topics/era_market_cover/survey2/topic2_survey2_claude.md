# SURVEY_metrics: quantitative evaluation of era-targeted cover generation and stem-wise era/market analysis

Prepared 2026-10-25 for the CommE5070 final project. Companion to `SURVEY.md`. Sources were located and checked during this survey unless listed under "Unverified or uncertain". **Reported** = stated by the cited source. **Interpretation** / **my design** = my own reasoning, not established practice.

---

## 0. Diagnosis of your three problems (read first)

1. **"Every generated clip is predicted as the 2000s."** This is the expected behaviour of a probe trained on real audio and tested on decoder output. Generated music carries decoder artefacts that are so easy to detect that a simple CNN reaches 99.8% accuracy telling real audio from its autoencoder reconstruction, controlling for musical content and file encoding ([Afchar et al., arXiv:2405.04181](https://arxiv.org/abs/2405.04181)). Your era probe is partly reading "decoder fingerprint", and the 2000s class is where that fingerprint lands. Fix: put the **reference side through the same codec** and score **relative** to a neutral-prompt baseline (Section 2).
2. **Encoders disagree; MuQ moves under a crude DSP filter, MERT does not.** This means at least one probe reads production cues. That is not a bug of the metric to hide; it is a finding to report, and the DSP filter becomes a standing **counterfactual control** (Section 7).
3. **Chroma cosine floor 0.63 to 0.77 for a different song.** Mean-chroma and frame cosine reward any tonal music in a related key. Fix: stop reporting the raw similarity and report a **retrieval/identification score** (does the remix match its own original better than the 11 other originals?) plus a **transposition-rotation floor**. Rotation-based chance levels are an established trick ([pitch-class steering paper, arXiv:2609.04516](https://arxiv.org/abs/2609.04516)).

Small n (12 songs) means you can only detect large effects; see the power table in Section 8.

---

## 1. Era or style presence in generated music

| Metric family | What it measures | Reference data | Small-n behaviour | Known failure modes | Code | Human-correlation evidence | Usable at 12 songs × 3 eras × 2 to 3 seeds? |
|---|---|---|---|---|---|---|---|
| **Classifier/probe score** (P(era = X), expected decade) | Whether a trained era classifier assigns the target class | Labelled real era audio (your accompaniment stems) | Per-item, so fine at small n | Domain shift to generated audio (above); reads production/codec cues; circular if same probe guided generation | Your probes | None for era | Only in relative, codec-matched form (Section 2) |
| **FAD** ([Kilgour et al., Interspeech 2019](https://www.isca-speech.org/archive/interspeech_2019/kilgour19_interspeech.html)) per era | Distance between generated set and real era-X set in an embedding space | ≥ hundreds of real era clips per era | **Reported biased at small sample size**; FAD∞ extrapolation proposed to reduce it ([Gui et al., arXiv:2311.01616](https://arxiv.org/abs/2311.01616)) | Embedding choice dominates: in environmental audio, VGGish FAD correlated below 0.1 with ratings while PANNs reached Spearman > 0.5, and music embeddings did worse there ([Tailleur et al., arXiv:2403.17508](https://arxiv.org/abs/2403.17508)) | [fadtk](https://github.com/microsoft/fadtk) | Gui et al. report correlation with MusicCaps labels and a listening test for good embedding/reference choices | Weak: n per era ≈ 24 to 36 clips; use only in relative form |
| **KAD** (MMD-based) ([Chung et al., arXiv:2502.15602](https://arxiv.org/abs/2502.15602), ICML 2025) | Same idea, distribution-free, unbiased | Same | **Reported faster convergence at small sample sizes**, unbiased estimator | Still embedding-dependent | [kadtk](https://pypi.org/project/kadtk/) (supports CLAP, MERT, PANNs, EnCodec embeddings, among others) | Authors report stronger alignment with human perception than FAD | **Better than FAD for you**; use the own-era margin (Section 9) |
| **Per-song FAD** (fadtk `--indiv`) | Distance of one clip to a reference set | Reference set | Per item | Same as FAD | fadtk | Gui et al. report per-song FAD can flag outliers and predict perceptual quality | Possible as a per-song era score, untested for era |
| **kNN era vote / nearest real-era neighbour** | Fraction of k nearest real clips (in an embedding) that are era X | Labelled real clips | Per item | Same domain shift as the probe unless references are codec-matched | sklearn | None found for era | Yes, with codec-matched references; a useful second opinion to the probe |
| **CLAP text-audio score** ("a song from the 1970s") ([LAION-CLAP, arXiv:2211.06687](https://arxiv.org/abs/2211.06687)) | Text-audio cosine | None | Per item | **I found no validation of CLAP for era/decade alignment.** CLAP zero-shot on music tasks is weak in general benchmarks | LAION-CLAP | None for era | Secondary only; must pass the positive control in Section 7 before use |
| **Discogs style taggers** (Essentia Discogs-EffNet, MAEST; 400 Discogs styles) ([Essentia models](https://essentia.upf.edu/models.html)) | Style posteriors (e.g. "Disco", "Synth-pop", "Doo Wop") | None at inference; you map styles → era | Per item | Styles are not decades; MAEST's training data is reported as over-representing Western electronic music ([model card](https://huggingface.co/mtg-upf/discogs-maest-30s-pw-73e-ts)); same codec shift | essentia-tensorflow, HF | None for era | Yes as an **independent** era evaluator (different training data from your probe); build the style→era map from your real data, not by hand |
| **Production features** (LUFS, crest factor/dynamics, spectral roll-off/bandwidth, stereo width, noise floor) | Production profile | Era distributions from your real clips | Per item, cheap | These are exactly the shortcut cues; a DSP filter can move them without changing style. Era trends reported: loudness increase and dynamics tied to year ([Deruty & Pachet, ISMIR 2015](https://archives.ismir.net/ismir2015/paper/000136.pdf)) | pyloudnorm, librosa | Not for era perception | Report as a **shortcut detector**, not as success |
| **Image/audio style-transfer metrics** (Gram-matrix distances, ArtFID-like content+style combinations) | Texture statistics | Style reference | — | Designed for images; no music validation found | — | — | Not recommended (interpretation) |

**Interpretation.** No era metric is validated against human judgement in the literature I found. Your era metric therefore has to be validated by you with controls (Section 7). The defensible choice at your n is **a relative, codec-matched, two-encoder probe score**, cross-checked by an **independent style tagger** and **KAD own-era margin**.

---

## 2. Fixing the domain-shift problem

| Approach | Status | Notes |
|---|---|---|
| **Codec-matched references**: pass real era recordings through the generator's codec (MusicGen uses a 32 kHz EnCodec tokenizer, [MusicGen paper](https://arxiv.org/abs/2306.05284)) before training the probe or building reference sets | **Established in a neighbouring field**: Afchar et al. auto-encode real music so that real and fake differ only in decoder artefacts ([arXiv:2405.04181](https://arxiv.org/abs/2405.04181)). Using it to *equalise* artefacts for an era evaluator is **my design** | Cheapest high-value fix. Train/evaluate the era probe on EnCodec-resynthesised real accompaniment; generated audio then sits in-domain for codec artefacts. Note the same paper reports detectors generalise poorly to unseen autoencoders, so match the exact codec |
| **Neutral-prompt baseline / difference scores**: Δ = P(X \| prompt X) − P(X \| neutral prompt), same song, same seed | Common logic in controllable-generation ablations; as an era metric, **my design** | Cancels the "everything looks 2000s" offset. Report per song, averaged over seeds |
| **Difference-in-differences** across target eras: [P(X\|prompt X) − P(X\|prompt Y)] for X≠Y | **My design** | Removes per-song biases; one number per (song, era pair) |
| **Ordinal ordering score**: Spearman between prompted decade and the probe's expected decade within each song | **My design** | Directly tests your MuQ ordering problem (1980s prompt predicted older than 1960s) |
| **DSP-only counterfactual** (band-limit + mono + saturation + lower loudness) | Logic of interventions is established in MIR ([Sturm 2014](https://doi.org/10.1109/TMM.2014.2330697); [Rodríguez-Algarra et al. 2019](https://doi.org/10.5334/tismir.24)); applied here, **my design** | Report generation's era score *relative to* the DSP filter; an evaluator that rates DSP-only as "1960s" as strongly as generation is reading production |
| **Production-invariant evaluation**: equalise bandwidth, mono, LUFS on *all* audio before embedding | **My design** (same as SURVEY.md §3) | Report era scores with and without; the gap is the production contribution |
| **Training the probe on generated audio of known era** | Not established; needs generated audio with ground-truth era, which you do not have | Skip |
| **Adversarial / domain-adversarial probe** | Exists generally in ML; nothing music-specific found | Too costly for one week |

---

## 3. Content and melody preservation

### 3.1 What the reference papers report (exact metrics)

| Paper | Melody/content metric (as reported) |
|---|---|
| MusicGen-Melody ([arXiv:2306.05284](https://arxiv.org/abs/2306.05284)) | **Chroma cosine similarity**: average cosine between frames at the same time steps of the quantised chroma of reference and generated audio; plus a human melody-match rating on 1 to 100 |
| Music ControlNet ([arXiv:2311.07069](https://arxiv.org/abs/2311.07069)) | Melody control signal = frame-wise argmax of a librosa chromagram after a high-pass at middle C (≈261 Hz) to avoid bass dominance; melody faithfulness evaluated against this representation (reported 49% more faithful than MusicGen) |
| SAO ControlNet ([arXiv:2410.05151](https://arxiv.org/abs/2410.05151)) | Top-k CQT melody representation (high-pass at middle C, 4 most prominent pitches per channel), argued to be less ambiguous than one-hot chroma |
| MuseControlLite ([arXiv:2506.18729](https://arxiv.org/abs/2506.18729)) | **Melody accuracy**: chromagram (STFT window 2048, hop 512), argmax pitch class per frame, frame-wise agreement between reference and generated; code taken from SAO ControlNet authors. Reported 61.1% vs 56.6% (SAO ControlNet) vs 43.1% (MusicGen-Stereo-Large-Melody) |
| SongEcho ([arXiv:2602.19976](https://arxiv.org/abs/2602.19976)) | RMVPE vocal F0 at 100 Hz, then `mir_eval` **Raw Pitch Accuracy, Raw Chroma Accuracy, Overall Accuracy**; plus FD_openl3, KL_passt, CLAP, phoneme error rate via Whisper |
| MuseCPEval ([arXiv:2512.14629](https://arxiv.org/abs/2512.14629), ISMIR 2026; [code](https://github.com/Yashvishe13/MuseCPEval)) | A full "music context preservation" suite: circle-of-fifths key distance, mean-chroma cosine, chroma DTW similarity, folded ΔBPM, beat F-measure and information gain (`mir_eval`), structural pairwise F and ARI (msaf), contour DTW similarity, motif 3-gram recall; validated with known symbolic edits and an 11-listener test |

**Important for your floor problem (reported by MuseCPEval):** under known edits on rendered MIDI, mean-chroma similarity stays ≈0.99 for structure, tempo and melody-inversion edits, and is still 0.82 to 0.94 after transposition; chroma DTW stays 0.71 to 0.96. These metrics are insensitive by construction, which matches your 0.63 to 0.88 floor.

### 3.2 Options and their floors

| Metric | Floor / ceiling issue | How to report (interpretation) | Tool |
|---|---|---|---|
| Mean-chroma cosine | Very high floor (above) | Do not use as primary | librosa |
| Frame chroma / CQT cosine, DTW-aligned | High floor; DTW raises it further (your 0.78 to 0.88) | Report only as **lift over a song-shuffled floor** and over **11 transposition rotations** of the reference | librosa, `librosa.sequence.dtw` |
| Argmax pitch-class agreement (MuseControlLite metric) | Chance is not 1/12; measure empirically by rotating the reference pitch class (rotation control, [arXiv:2609.04516](https://arxiv.org/abs/2609.04516)) | Accuracy minus rotation-chance | librosa |
| **Identification / retrieval score**: for each generated clip, rank its own original among all 12 originals by similarity (chroma-DTW or a learned embedding) | Floor = 1/12 top-1, MRR baseline ≈ 0.26 for 12 items (interpretation) | **Recommended primary**: top-1 accuracy and mean reciprocal rank, CI by bootstrap over songs | numpy |
| Melody F0 accuracy (RMVPE, [arXiv:2306.15412](https://arxiv.org/abs/2306.15412); CREPE; basic-pitch [arXiv:2203.09893](https://arxiv.org/abs/2203.09893)), `mir_eval.melody` RPA/RCA/OA | Trivially ≈ original on the full remix because you keep the vocal stem | Only meaningful if you measure the **lead line in the generated accompaniment** against the original accompaniment's lead (instrumental melody) | mir_eval |
| Chord-sequence agreement (chord recogniser on original vs generated accompaniment; `mir_eval.chord` weighted overlap at "majmin" level) | Chance depends on key; use the rotation floor | Good harmony check; harmony is what MusicGen-Melody's chroma should keep | madmom or similar recogniser + mir_eval (see Unverified for the recogniser choice) |
| Beat/tempo agreement (folded ΔBPM; beat F-measure ±70 ms) | Generation is not time-locked, so beat F can collapse while tempo is fine (MuseCPEval reports this for edits) | Report folded ΔBPM primarily; beat F secondary | Beat This! ([arXiv:2407.21658](https://arxiv.org/abs/2407.21658)) + mir_eval |
| Key agreement (circle-of-fifths distance) | Coarse | Secondary | essentia key or MuseCPEval code |
| Cover-song ID embedding distance (CLEWS [arXiv:2502.16936](https://arxiv.org/abs/2502.16936); CoverHunter [arXiv:2306.09025](https://arxiv.org/abs/2306.09025); ByteCover) | Needs calibration against real versions (you do not have them yet) | If used: report as identification rank among the 12 originals (same as the retrieval score above), which needs no calibration set | CLEWS checkpoints on Zenodo (GPU) |

---

## 4. Audio quality and prompt adherence (secondary checks)

| Metric | Notes for tiny n | Tool |
|---|---|---|
| **Audiobox Aesthetics** (PQ production quality, PC production complexity, CE content enjoyment, CU content usefulness) ([arXiv:2502.05139](https://arxiv.org/abs/2502.05139)) | **No reference set needed, per item** → best fit for n = 12. Training data included generated music (PAM set). Reported comparable or better correlation with MOS than prior predictors | `pip install audiobox_aesthetics` ([repo](https://github.com/facebookresearch/audiobox-aesthetics)); GPU optional |
| PAM (prompting CLAP for quality) ([arXiv:2402.00282](https://arxiv.org/abs/2402.00282)) | Per item, no reference; reported to correlate with human scores on text-to-music | MS-CLAP based |
| KAD / FAD vs real accompaniment | Set-level; FAD biased at small n; use KAD | kadtk, fadtk |
| KL on tagger outputs (KL_passt) | Needs paired reference; standard in MusicGen/SongEcho tables | stable-audio-metrics |
| CLAP score (text adherence) | Per item; era text alignment unvalidated (Section 1) | LAION-CLAP |
| Caveat | Vinay & Lerch report that commonly used objective metrics were insufficient to describe perceptual quality of generative audio systems ([arXiv:2209.00130](https://arxiv.org/abs/2209.00130)) | — |

---

## 5. Vocal and remix consistency

| Check | Metric | Source / status |
|---|---|---|
| **Does the new accompaniment fit the kept vocal?** | **COCOLA score**: contrastive model of harmonic and rhythmic coherence between stems, proposed specifically to evaluate accompaniment generation; checkpoints trained on MUSDB18-HQ, MoisesDB, Slakh2100, CocoChorales | [arXiv:2404.16969](https://arxiv.org/abs/2404.16969), [code](https://github.com/gladia-research-group/cocola). Report against two controls (my design): original vocal + original accompaniment (ceiling) and vocal + another song's generated accompaniment (floor) |
| Key match | Pitch-class histogram of the vocal F0 (RMVPE) vs accompaniment chroma; correlation or circle-of-fifths distance | Interpretation |
| Tempo / timing alignment | Folded ΔBPM between vocal-onset tempo and accompaniment beats; fraction of vocal onsets within ±70 ms of an accompaniment beat subdivision | Interpretation; tools Beat This!, librosa onsets |
| Balance | Vocal-to-accompaniment loudness ratio (LUFS difference) vs original | pyloudnorm; interpretation |
| Masking | No standard metric found for music remixes; skip | — |

---

## 6. Metrics for the analysis half

| Construct | Recommended metric | Notes / source |
|---|---|---|
| Overall accuracy | Balanced accuracy and macro-F1 | Standard |
| Decade is ordinal | **Mean absolute decade error**, within-one accuracy, **quadratic weighted kappa** (`sklearn.metrics.cohen_kappa_score(weights="quadratic")`) | Standard; QWK and MAE penalise 1960s→2010s more than 1960s→1970s |
| Probabilistic quality | **Log-loss** (primary), Brier; ECE with few bins only | ECE is noisy at n ≈ 100 to 130 (interpretation) |
| Stem attribution | Accuracy difference mix vs vocals vs accompaniment vs recombined, **exact McNemar** on the same items; **two-player Shapley** with v(S) = probe accuracy on concatenated embeddings of stems in S, v(∅) = chance | Shapley with 2 stems is exact and needs 4 probes (interpretation). SHAP background: [Lundberg & Lee, NeurIPS 2017](https://arxiv.org/abs/1705.07874) |
| How easily the information is extracted | **MDL probing (online code length)** | [Voita & Titov, EMNLP 2020](https://arxiv.org/abs/2003.12298), [code](https://github.com/lena-voita/description-length-probing). Reported to be more informative and stable than probe accuracy |
| Probe expressivity control | **Control task + selectivity**: assign random labels per *artist* (your analogue of word type) and report task accuracy minus control accuracy | [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368) as described by Voita & Titov; the per-artist variant is my design and directly measures artist-identity shortcuts |
| Uncertainty | Bootstrap CIs resampling **artists** (cluster bootstrap); Bayesian correlated t-test / signed-rank via [baycomp](https://pypi.org/project/baycomp/) ([Benavoli et al., JMLR 2017](https://www.jmlr.org/papers/v18/16-305.html)) | |
| Multiple comparisons | Holm across the family of stem comparisons per task | Standard |

**Small standard set (recommendation):** balanced accuracy, MAE-decade, QWK, log-loss, exact McNemar mix-vs-stem with Holm, artist-cluster bootstrap CIs, plus MDL as the one "beyond accuracy" attribution number.

---

## 7. Metric validation plan without listeners

**How papers validate metrics (reported).** Correlation with human ratings (Gui et al. 2024; Tailleur et al. 2024; Audiobox Aesthetics), sensitivity to known manipulations with pre-stated expected values (MuseCPEval's symbolic edits with an "Expected" column), and rotation/shuffle chance controls ([arXiv:2609.04516](https://arxiv.org/abs/2609.04516)).

**Your plan (my design, ~2 person-days):**

| Control | Construction | Expected behaviour of a valid era metric | Expected behaviour of a valid content metric |
|---|---|---|---|
| (a1) Positive, codec | Real held-out era-X accompaniment passed through EnCodec | Recovers X; decade ordering Spearman high | — |
| (a2) Positive, oracle | Replace generated accompaniment with a *real* era-X accompaniment of another song | Recovers X | Content score drops to the shuffled floor |
| (b1) Negative, neutral | Neutral-prompt generation | Δ ≈ 0 by definition | Content same as era prompts |
| (b2) Negative, shuffled references | Swap era labels of the reference set | Margin ≈ 0 | — |
| (b3) Negative, noise | Pink noise / random audio | No era preference; quality metric low | Retrieval at chance |
| (c) Counterfactual DSP | Band-limit + mono + saturation + lower LUFS on the *original* accompaniment | Should move less than generation; report the ratio | Content unchanged (must stay high) |
| (d) Sensitivity curve | Linear crossfade from original accompaniment (α = 0) to generated (α = 1), 5 steps | Monotone change in Δ | Monotone decrease |
| (e) Seed variance | 2 to 3 seeds per (song, prompt) | Intra-class correlation across seeds reported | Same |
| (f) Encoder agreement | Same metric with MERT and MuQ (and a tagger) | Same sign of Δ per era | — |

**Acceptance criteria (my proposal; state them before running):**
1. Positive control beats its matching negative control with a 95% bootstrap CI excluding zero.
2. Decade-ordering Spearman ≥ 0.5 on the codec positive control (a1).
3. Sensitivity curve monotone (Spearman with α ≥ 0.8 averaged over songs).
4. Seed ICC ≥ 0.5; otherwise average more seeds before testing.
5. Two encoders agree in sign of Δ for each era; if not, report the metric as encoder-dependent, not as a result.
6. Content metric: retrieval top-1 on the oracle-swap control at or near chance, and on DSP control near 1.

---

## 8. Statistics for small n

**Design.** Unit of analysis = song. Average seeds within (song, condition) first, then test across songs. Seeds are not independent replicates of the hypothesis.

**Tests.**
- Primary: **exact paired sign-flip permutation test** (or Wilcoxon signed-rank) on per-song differences; report the mean difference and Hodges–Lehmann estimate with a **bootstrap CI over songs**. With 12 songs the smallest attainable two-sided exact p is 2/2^12 ≈ 0.0005 (computed).
- Robustness: linear mixed model with random intercept for song (and seed nested), `statsmodels` MixedLM; report if conclusions differ.
- Multiplicity: Holm over the 3 target eras (and over metrics if several are primary; better: one pre-declared primary per construct).
- Bayesian alternative: baycomp signed-rank with a region of practical equivalence.

**Power (my Monte Carlo simulation, paired t-test, normal differences, 4,000 replications; d = mean/SD of per-song differences):**

| Songs | α | d = 0.5 | d = 0.8 | d = 1.0 | d = 1.2 |
|---|---|---|---|---|---|
| 12 | 0.05 | 0.35 | 0.70 | 0.88 | 0.96 |
| 12 | 0.05/3 (Holm worst case) | 0.18 | 0.51 | 0.72 | 0.89 |
| 20 | 0.05 | 0.56 | 0.93 | 0.99 | 1.00 |
| 20 | 0.05/3 | 0.37 | 0.82 | 0.97 | 1.00 |
| 30 | 0.05 | 0.76 | 0.99 | 1.00 | 1.00 |

Wilcoxon power at d = 0.8 was 0.66 (n = 12) and 0.91 (n = 20), similar to the t-test. Interpretation: 12 songs reliably detect only d ≈ 1 or larger after correction. **Adding songs helps far more than adding seeds**; going from 12 to 20 songs is the single best investment.

**Guidance from the literature.** Vinay & Lerch cite statistical-testing guidance for MUSHRA-type data and show objective metrics are often insufficient ([arXiv:2209.00130](https://arxiv.org/abs/2209.00130)); the Lerch et al. survey on evaluating generative music models reviews practice ([arXiv:2506.05104](https://arxiv.org/abs/2506.05104)). I found no music-generation paper that gives a power analysis for paired song-level designs.

---

## 9. Recommended metric suite

| Construct | Primary (one) | Secondary (1 to 2) | Validation from §7 |
|---|---|---|---|
| **Era presence** | **Codec-matched relative probe score**: Δ_X = P(X \| prompt X) − P(X \| neutral), probe trained on EnCodec-resynthesised real accompaniment, reported for MERT and MuQ | (1) Discogs-style tagger era score (MAEST/EffNet styles mapped to decades on real data), same Δ form. (2) **KAD own-era margin**: KAD(gen_X, ref_other eras) − KAD(gen_X, ref_X) with codec-matched references | a1, a2, b1, b2, c, d, e, f |
| **Content preservation** | **Song-identification score**: top-1 and MRR of each generated accompaniment against the 12 original accompaniments, chroma-DTW similarity, high-passed at middle C | (1) Chord agreement (majmin, mir_eval) minus rotation floor. (2) Folded ΔBPM | a2, b3, c, d |
| **Quality** | **Audiobox Aesthetics PQ and CE** (per item) | KAD vs codec-matched real accompaniment | b3, e |
| **Remix fit** | **COCOLA** vocal↔new accompaniment, vs original pairing (ceiling) and cross-song pairing (floor) | Key match (vocal F0 histogram vs accompaniment chroma) | ceiling/floor built in |
| **Analysis attribution** | Balanced accuracy per input with exact McNemar (Holm) | MAE-decade + QWK; MDL online code length; per-artist control-task selectivity | Artist-cluster bootstrap |

**Effort (person-days, interpretation):**

| Item | Effort | GPU? | Libraries |
|---|---|---|---|
| EnCodec resynthesis of real era accompaniment + re-embed + retrain probes | 1.0 | Yes (embedding, codec) | audiocraft, your probe code |
| Δ / DiD / ordering scores, tagger era score | 0.5 | Tagger: optional | essentia-tensorflow or HF MAEST |
| KAD margins | 0.5 | Optional | kadtk |
| Song-identification + chord + tempo | 1.0 | Beat This!: optional | librosa, mir_eval, madmom (chords), beat_this |
| Audiobox Aesthetics, COCOLA | 0.5 | Yes recommended | audiobox_aesthetics, cocola |
| Controls a to f | 1.5 | Some | pyloudnorm, scipy, sox-style DSP in numpy |
| Analysis metrics + MDL + stats | 1.0 | No | scikit-learn, statsmodels, baycomp, scipy.stats.permutation_test |
| **Total** | **≈ 6 person-days** | | |

**Qualitative listening round (kept separate, at the end).** Check: (1) whether high-Δ items actually sound like the target era or only "older/duller" (the DSP failure mode); (2) audible decoder artefacts that the quality predictor missed; (3) vocal–accompaniment clashes where COCOLA was high; (4) the extreme items of each metric (top and bottom 2 per metric) rather than random items. Report as observations, not as a score.

---

## 10. Reading list (in order)

1. Afchar et al. 2024, "Detecting music deepfakes is easy but actually hard" [arXiv:2405.04181](https://arxiv.org/abs/2405.04181): why your probe sees "2000s" everywhere, and the auto-encoding trick to cancel it.
2. MuseCPEval 2026 [arXiv:2512.14629](https://arxiv.org/abs/2512.14629): a full content-preservation suite with known-edit validation you can copy.
3. Chung et al. 2025, KAD [arXiv:2502.15602](https://arxiv.org/abs/2502.15602): the distribution metric that behaves at small n.
4. Gui et al. 2024, Adapting FAD [arXiv:2311.01616](https://arxiv.org/abs/2311.01616): FAD bias and embedding choice.
5. Tailleur et al. 2024 [arXiv:2403.17508](https://arxiv.org/abs/2403.17508): embedding choice can make FAD meaningless.
6. Tsai et al. 2025, MuseControlLite [arXiv:2506.18729](https://arxiv.org/abs/2506.18729) §4.4: the melody-accuracy definition used in recent work.
7. Ciranni et al., COCOLA [arXiv:2404.16969](https://arxiv.org/abs/2404.16969): measuring whether an accompaniment fits a given stem.
8. Audiobox Aesthetics [arXiv:2502.05139](https://arxiv.org/abs/2502.05139): reference-free per-item quality.
9. Voita & Titov 2020, MDL probing [arXiv:2003.12298](https://arxiv.org/abs/2003.12298): attribution beyond probe accuracy, with control tasks.
10. Benavoli et al. 2017 [JMLR](https://www.jmlr.org/papers/v18/16-305.html): Bayesian comparison of classifiers for small validation sets.

---

## Unverified or uncertain

- EnCodec paper id and the exact audiocraft API for encode/decode resynthesis with MusicGen's 32 kHz compression model: not checked in this survey.
- Hewitt & Liang 2019 arXiv id 1909.03368: taken from memory; the paper itself is confirmed through Voita & Titov's description.
- Lundberg & Lee SHAP arXiv id 1705.07874, Guo et al. ECE paper, Dietterich 1998 McNemar recommendation: not re-verified.
- Chord recogniser choice (madmom CNN/CRF chord models): availability and licence not checked; madmom's DBN components are reported elsewhere as non-commercial.
- Music ControlNet's exact evaluation formula for melody accuracy: control representation verified; that its metric is frame-wise argmax agreement is inferred from MuseControlLite reusing SAO ControlNet's code.
- CREPE paper id and RMVPE's exact accuracy on separated vocals: not checked.
- CLAP era/decade alignment: searched and found nothing; absence of evidence, not proof.
- Discogs style→era mapping quality: untested; build it on your own data.
- COCOLA on vocal stems specifically: checkpoints are trained on datasets with vocals (MUSDB18-HQ, MoisesDB), but performance on vocal↔accompaniment pairs was not checked.
- Power numbers in §8 come from my simulation under normal per-song differences; real metric distributions may be skewed.
- Acceptance thresholds in §7 are my proposals, not literature standards.
