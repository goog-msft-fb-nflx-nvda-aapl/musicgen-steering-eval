# plan

topic: independent, statistically-confirmed evaluation of concept-direction steering in MusicGen.
chosen 2026-10-05 from docs/response.md (deep-research reply to docs/RESEARCH_PROMPT.md), with scope changes below.

## questions (all open, no expected outcome assumed)

- Q1: does steering success hold up under evaluators independent of the steering probe (dsp: chroma/essentia/onset rate; frozen mert/muq probes), vs the self-probe numbers musicrfm reports? report the gap.
- Q2: four direction methods behind one interface (mean-difference/caa, logistic-probe normal, lda, rfm top eigenvector), compared at matched quality cost (control accuracy vs mad/clap pareto front), not at one shared eta.
- Q3: is the layer where a concept is best decodable also where steering works best? layer chosen by probe auc vs layer chosen by measured steering effect on dev prompts, evaluated on held-out prompts (out-of-fold, same discipline as pa1 fusion weights).

## update 2026-10-14 after reading musicrfm and neighbours (details in SURVEY.md)

- Q1 reworded in practice: musicrfm already reports chroma/essentia/onset checks, so a chroma-argmax evaluator alone replicates them. our additions are paired design, significance tests and a validated independent evaluator. do not claim "circular evaluation" as their flaw.
- new arm: faithful musicrfm config (exp layer weighting kappa 0.95, injection p=0.3, eta0 0.15-0.6) next to my constant-injection top-8 config, since my pilot at eta>=1.0 used p=1.
- direction methods: musicrfm has no mean-diff/logistic/lda baselines, panda et al only for genre/timbre. that gap is Q2.
- Q3 has non-music prior art (2609.22135, 2604.15557); frame as testing it on a music generator.
- optional extension from musicrfm future work: higher rfm eigen-components (already trained, n_components=12).

## scope changes vs response.md

- main grid on musicgen-small only (full factorial). response proposed small+medium; medium dropped unless time is left.
- musicgen-large = confirmation subset only, since musicrfm validated on large. measure large throughput in the pilot before sizing it.
- concepts: start with notes + tempo + chord quality; drop any whose independent evaluator fails validation.
- 3 concepts x 4 methods x 4 strengths x ~100 paired prompt/seed clips x 10 s = ~4800 clips on small. per-layer causal sweep (dev prompts) is on top of this; size it after pilot throughput is known.
- title is a question, not "decodable != steerable": that is the hypothesis, not a finding.

## open items to verify before trusting the plan

- musicrfm repo (github.com/astradzhao/music-rfm) actually runs end to end; xrfm lib installs without touching shared envs (make a new conda env, do not pip into rl_hw3).
- "real labeled audio" for the independent mert/muq probes: response.md asserts it, but notes have no natural real-audio set. decide per concept: tempo (real tempo-annotated set), chords (check dataset licence), notes (dsp only, or probe trained on held-out synthetic). state which evaluators are truly independent of synthetic training data.
- novelty list in response.md (six prior steering papers) is unverified; read the closest ones before the pitch.
- musicgen weights cc-by-nc 4.0: fine for course use, do not redistribute weights.

## pilot (days 1-3)

one note direction, musicgen-small, 20 prompts x 12 target notes, eta in {0, 0.3, 0.6}. rfm-probe accuracy vs chroma-argmax accuracy, paired by seed. also record gen throughput small/large.

## milestones (gpu ends 2026-10-31)

- w1 (10/05-10/11): env + repo running on gsm-gpu2, activations on synctheory, pilot.
- w2 (10/12-10/18): four direction methods, validated evaluators, per-layer sweep on dev prompts.
- w3 (10/19-10/25): full held-out grid on small. save all audio + metadata.
- w4 (10/26-10/31): large confirmation subset, demo clip selection, extract ALL mert/clap embeddings before the cutoff.
- nov (cpu only): stats (paired bootstrap, mcnemar, wilcoxon, holm), pareto plots, online paired a/b listening test, report.

## compute rules

gsm-gpu2 only gpu 1 or 2 (0 and 3 broken), check all indices, smoke-test with a real .cuda() op. one experiment at a time. tmux. cap omp/mkl threads. delete /tmp scratch after each cycle. server dir: /home/jtan/musicgen_steer/.
