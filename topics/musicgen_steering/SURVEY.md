# survey (read 2026-10-14; abstracts/html via web fetch, not full careful reads)

only what was actually read is claimed here. anything marked (unverified) came from a search snippet.

## closest prior work, music

- musicrfm, zhao et al, iclr 2026, arxiv 2510.19127 (v2 apr 2026), code github.com/astradzhao/music-rfm.
  - steers musicgen-large (48 layers) with top rfm eigenvector from probes on mean-pooled synctheory hidden states.
  - layer choice: per-layer probe val auc (or mse for regression); main results use exponential layer weighting w = w0 * s^(1/kappa), w0=1, kappa=0.95; top-k ablated k=4..48. naive all-48-layer steering degrades quality.
  - eta0 in {0.15, 0.30, 0.45, 0.60}; injection probability p=0.3 for main results (ablated 0.15..1.0).
  - metrics: fd and mmd vs unsteered large, clap (630k-audioset-fusion-best), multiclass rfm probe accuracy ("relative indicators", may not transfer to real generations).
  - they DO have a non-probe check for some concepts: chroma argmax for notes, essentia chords, librosa onset rate for tempo (spearman with eta0 0.283 rfm, 0.433 rfm+prompt). 12-person listening test, no significance test reported.
  - NOT done by them: no comparison with mean-difference / logistic / pca directions (logistic mentioned only as failing to converge), only musicgen-large, only top eigenvector, mean pooling only.
  - their stated future work: temporal pooling / sequence-level rfm, multiple eigen-components, masking ar context, timbre/instrument/articulation, jukebox and magenta-rt.
- panda et al, arxiv 2506.10225: musicgen-melody residual stream and attention outputs; directions from difference-in-means or probe weights (mse-trained probes beat cross-entropy ones, their claim); evaluation by clap zero-shot genre change (independent of the probe) on few tracks; strength by hyperparameter search (they flag as unreliable); superposition not understood.
- activation patching for steering in music, arxiv 2504.04479: difference-in-means vectors for binary concepts (tempo, timbre).
- singh et al, sparse autoencoders on musicgen residual stream, iclr 2026, arxiv 2505.18186 (abstract level only).

## closest prior work on decodable-vs-steerable (non-music)

- "read-best is not steer-best", arxiv 2609.22135: three omni-modal llms, emotion; probe-best layer varies widely, steer-best layer sits in a narrow mid-to-late band of relative depth; paired random-direction controls (about 26x causal gap reported). audio is an input modality there; no music generation.
- "predicting where steering vectors succeed", arxiv 2604.15557: training-free logit-lens "linear accessibility profile" predicts steering success (spearman 0.86-0.91 across 24 concepts, 5 text llms).
- layer-selection-by-probe-accuracy criticized as unstable under probe-set resampling (search snippet only, unverified).

## what this means for our plan (interpretation)

- response.md's "circular evaluation" framing of musicrfm is too strong: they already use chroma/essentia/onset checks for notes/chords/tempo. our chroma-argmax note evaluator is the same family as theirs, so it is a replication-level check, not a new independent evaluator. what stays open: significance testing, paired design, fidelity-matched comparison, and a validated probe-independent evaluator on real audio.
- direction-method comparison (our Q2) is partly covered for genre/timbre by panda et al (probe weights vs diff-in-means), but not for music-theoretic concepts (notes/chords/tempo) and not against rfm. musicrfm has no such baseline.
- decodable vs steerable (our Q3) is studied in omni-modal llms and text llms; no music-generation version found yet. still open for musicgen, not a novelty claim for the idea in general.
- candidate additions that match stated future work and are cheap: (a) use higher rfm components (we already train n_components=12), (b) the faithful musicrfm config as a baseline arm (exp layer weighting kappa .95, p=0.3, eta0 .15-.6), (c) a logit-lens-style predictor of steerable layers adapted to musicgen's codebook heads (speculative, not tested, analogy to 2604.15557).

## to read properly before the pitch
- 2609.22135 and 2604.15557 full text; 2504.04479; musicrfm appendix on listening test and tempo.
