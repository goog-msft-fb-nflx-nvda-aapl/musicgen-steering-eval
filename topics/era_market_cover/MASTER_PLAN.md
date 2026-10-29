# Master plan: sources, component choices, ablation registry, next steps (topic 2)

Status date: 2026-10-28. Companion files: `WORKLOG.md` (every measured result, with config), `TODO.md` (short list), `TA_COVERAGE.md` (TA's questions vs evidence), `SURVEY3.md` (verified citations), `repro/environment.txt` (versions and hashes), `COURSE_MATERIALS_NOTES.md`.
Rules for this document: only measured items are called done; "proxy", "interpretation" and "unverified" are kept as such; every experiment has an ID so WORKLOG, report and slides can cite it; one experiment at a time on GPU 1 or 2 (GPU 0/3 are broken on the server).

## 1. Where each idea came from (so a reviewer can trace every design choice)
| Source | What we took | Experiment IDs |
|---|---|---|
| Course lectures (2025 and 2026 editions) | `lecture11b` cover generation = content-preserving style transfer, evaluation critique; `DeepMIR-AutoMixing` (effects chain, FX normalization, Diff-MST, Fx-Encoder++); `lecture03b/04` separation (HT Demucs, BS-RoFormer, SDR caveat, HPSS); `lecture13` transcription (basic-pitch, chords); `lecture05/06` codecs (EnCodec) and diffusion + classifier-free guidance; `lecture03` CLAP/MuLan, MuseControlLite; `lecture02b` MERT/MuQ and the learnable layer-weighted-sum probe | A03, A08, A09, A06, G02, G10, E04, E06 |
| HW1 (PA1: Discogs-VI subset, frozen-embedding probes) | data, probe protocol (layer by CV, logistic regression, bootstrap CI, McNemar), fusion without leakage, zero-shot audio-language model with calibration | A01-A05, E08 |
| HW2 (PA2: POP909 symbolic generation) | training-seed replicates (`F1/FA12/F2`), CFG scale sweep, sampling parameter selection by entropy / perplexity matching (Compose & Embellish, lecture 05), learned reranker, result rows that carry config + code/checkpoint hashes, clean-environment inference test, upload package | G07, G09, G11b, R01-R04 |
| HW3 (PA3: text-to-music generation and evaluation) | CFG implementation (null branch vs conditional branch), metrics FD-VGG / KL-PaSST / CLAP (checkpoint `music_audioset_epoch_15_esc_90.14.pt`) and MQAScore (LALM as judge), improvement ideas on the slides: prompt engineering / recaptioning, CFG variants (composable / relative guidance), best-of-N + CLAP rerank, attention-map editing (MusicMagus) | G05, G06, G08, G09, E07, E08 |
| Survey 1, 2 | task framing, metric suite (KAD, Audiobox, COCOLA, chroma-DTW, chord, tempo), controls | A09, E01-E06 |
| Survey 3 (`SURVEY3.md`, verified items only) | closest prior work Lee et al. ISMIR 2026 (limitation = composition vs production); shuffle-and-remix (Hsieh 2020); LEACE (Belrose 2023); Demerlé 2024 (style-transfer covers); Discogs-VI metadata facts; "recalled" items (INLP, control tasks, cartography, MDL) marked in WORKLOG | A10-A14, A22 |
| TA description | all clauses; see `TA_COVERAGE.md` | all |

## 2. Component register: what we chose, what a reviewer will ask, the evidence
"Evidence" points to the registry in section 3. "Gap" means no ablation yet and the ID of the planned one.
| Component | Choice | Likely reviewer question | Evidence / gap |
|---|---|---|---|
| Data | PA1 course subset of Discogs-VI (30 s, 24 kHz, 6 decades; 6 markets in the 1980s), splits artist-disjoint by the organizers | Why not full Discogs-VI? Test labels? | Full Discogs-VI-YT needs YouTube audio (terms unclear; not run). Test labels hidden: all accuracy on validation (A 132, B 102). A22 gives label semantics of the full metadata |
| Evaluation of probes | validation accuracy once, CIs by bootstrap, McNemar, Holm; pooled 5-fold OOF for diagnostics | Variance / optimistic selection | A15 (3 CV seeds, Holm), pooled OOF is NOT artist-grouped (no artist ids): stated limitation |
| Stem separator | HTDemucs (primary), BS-RoFormer (check) | Separator artifacts? | A06, A07, A13 (both separators), cross-separator transfer shows vocal probes lose 0.08-0.13 |
| Loudness normalization | -23 LUFS | Does it matter? | A05 (raw vs norm rows) |
| Encoder | MERT-v2-30s (primary), MuQ, Whisper-medium enc., AST, CLAP | Why MERT? | A02 (MERT best or tied on every condition; CIs overlap) |
| Pooling over time | mean of hidden states | Mean pooling hides temporal cues | A17 (mean pooling best or tied in all six rows) |
| Layer choice | per-condition layer by 3-fold CV on train; also learnable layer-weighted sum | Selection on validation? | A03 (CV only; lws probe agrees), A04 |
| Probe | logistic regression, C=0.01 on standardized features | Capacity, regularization | A04 (SVM, MLP, concat, fusion: no clear gain); A18/A19 (C grid, learning curve: conclusions insensitive to C; accuracy not saturated for A) |
| Era generation backbone | MusicGen-Melody 1.5B (fp16), not claimed SOTA | Why this generator? | G12/G13 compare with MuseControlLite and ACE-Step 1.5; survey: no verified benchmark for era-controlled covers |
| What is regenerated | accompaniment only; original vocal kept; RMS-matched | Why not regenerate the vocal too? | By design (market lives in the vocals; content must stay). Remix level ablation G10 |
| Prompts | 4 hand-written era prompts; 5 extra prompts | Prompt sensitivity, prompt engineering | G04 (5 extra prompts); **Gap G05** (prompt elaboration), **Gap G06** (LLM / recaption prompts, HW3 idea) |
| Guidance | CFG scale 3 default; sweep 1, 3, 5, 7 | Why 3? | G02; G08 era-contrast (delta) guidance running |
| Sampling | transformers 5.x library defaults for do_sample=True: temperature 1.0, top_k 50, top_p 1.0 (checked in the library source; Meta's reference MusicGen uses top_k 250) | Entropy-matched sampling (HW2 idea) | G07 running (`68_sampling_params.py`: top_k 250, top_p 0.9, temperature 0.8 / 1.2) |
| Test-time selection | none yet | Best-of-N + independent reranker (HW3 idea) | **Gap G09** |
| Melody conditioning | chroma of the original mix | Is it needed? | G03 (text-only baseline: identity collapses, VINet top-1 .21 vs .72) |
| Era evaluator | codec-matched MERT L22 / MuQ L9 probes trained on EnCodec-resynthesized real accompaniment | Circularity, domain shift | E01 codec matching, E02 negative controls, E03 identity-codec rows; independent evaluators E04-E08 |
| Content metrics | chroma-DTW song-ID, chords, tempo, basic-pitch, Discogs-VINet | Validated against humans? | E09-E11 (automatic only); no human test (skipped by decision) |
| Distribution metric | KAD-style MMD margin (own implementation), Audiobox, COCOLA | Why not FAD? | E05; HW3 metrics FD-VGG / KL-PaSST not run (**Gap E07**) |
| LoRA fine-tune | r=16 attention q,k,v,out; 3000 steps, bs 8, lr 1e-4, 10% neutral-prompt dropout, eval mode | Rank, steps, seeds, targets | G11 (v1,v2,v3), G11b seeds running; **Gap G11c** (target modules, lr, dropout) |
| Language proxy | Whisper LID argmax on vocal stem | LID accuracy on singing | Proxy only; A24 (second LID model: same LEACE result); no ground-truth language exists (A23) |
| Style (genre) proxy | Discogs-EffNet 400 style tags on the original mix | Not ground-truth genre | A11, proxy named as proxy |
| Concept removal | LEACE (rank = groups), INLP, random same-rank controls | Linear only; correlated concepts | A10, A11; stated as upper bounds |
| Human evaluation | none | Perceived era / melody | Skipped by decision (limitation); LALM judge E08 is the training-free substitute |

## 3. Experiment registry
Legend: **done** = measured, logged in WORKLOG (section date), result file in `results/`; **run** = running now; **todo** = planned.
### A. Analysis half (what carries era and market)
| ID | Question | Status | Evidence |
|---|---|---|---|
| A01 | mix vs vocals vs accompaniment probes (MERT) | done | WORKLOG 2026-10-25 first results; `probe_{A,B}.json` |
| A02 | five encoders | done | `probe_encoders_*.json`, `probe_muq_*.json` |
| A03 | layer curves, CV selection, layer-weighted sum | done | `lws_*.json`, `figs/` |
| A04 | probe capacity (SVM, MLP, concat, fusion) | done | `probe_models_*.json` |
| A05 | raw vs loudness-normalized | done | `probe_*.json` (raw rows) |
| A06 | second separator | done | `probe_bsr_*.json`, cross-separator in `extra_controls_*.json` |
| A07 | vocal bleed | done | `bleed_summary.txt` |
| A08 | HPSS (harmonic vs percussive accompaniment) | done | `probe_hpss_*.json` |
| A09 | production-only controls (DSP, EnCodec round trip, EQ/compressor/reverb) | done | `controls_*.json`, `fx_*.json` |
| A10 | language: LID baseline, strata, English-only, LEACE, INLP | done | `leace_*.json`, `extra_controls_*.json`, `gap_B.*` |
| A11 | genre/style proxy: tags-only probe, clusters, LEACE erasure | done | `gap_*.json`, `style_erasure_*.json` |
| A12 | hard songs: clips wrong in all stems, cartography | done | `gap_*_clips.csv`, `extra_controls_*.json` |
| A13 | shuffle-and-remix (causal-style stem test), 2 separators | done | `shuffle_remix_*.json` |
| A14 | cue ablation: fixed probe and retrained probe | done | `cue_ablation_*.json`, `cue_retrain_*.json` |
| A15 | CV seeds, bootstrap, McNemar, Holm | done | `probe_cvseed*`, `holm_summary.txt` |
| A16 | sparse autoencoder on stem embeddings | done | `sae.json` |
| A17 | pooling / time resolution (mean, std, max, mean+std, mean+max, 3 segments) | done | `pooling_*.json` |
| A18 | probe regularization sensitivity table (C grid) | done | `probe_sensitivity_*.json` |
| A19 | learning curve (accuracy vs number of training clips) | done | `probe_sensitivity_*.json` |
| A20 | probe on pooled train+val with artist-agnostic repeated CV vs validation-only | done in part (gap analyses) | `gap_*.json` |
| A21 | artist-grouped bootstrap | impossible (no ids) | limitation |
| A22 | Discogs-VI metadata label semantics | done | `discogs_vi_meta.txt` |
| A23 | validate Whisper LID against ground-truth language | not possible (no language labels); agreement of two LID models reported in A24 | - |
| A24 | second language proxy (Whisper large-v3) and LEACE repeated | done | `leace_*_lid3.json` |
### B. Generation half (can era cues control a cover?)
| ID | Question | Status | Evidence |
|---|---|---|---|
| G01 | MusicGen-Melody era prompts, 24 songs x 3 seeds | done | `gen_eval_codec_24s3.json` |
| G02 | CFG scale 1, 3, 5, 7 | done | `cfg*_eval.json`, `cfg_summary.txt` |
| G03 | text-only (no melody) baseline | done | `textonly_*.json` |
| G04 | extra prompts (1970s, 1990s, 2000s, 1960s b/c) | done | `prompts_eval.json` |
| G05 | prompt elaboration (decade word / short / keyword tags vs the detailed sentence) | done | `src/65_prompt_variants.py`, `results/gens_summary_g08_g02b_g05_g07.txt` |
| G06 | LLM-written / recaptioned era prompts (HW3 slide 14), chosen on TRAIN-side songs, tested on the 24 val songs | done (4 prompt writers: Qwen3-8B, Qwen2.5-7B-Instruct, Qwen2.5-14B, Llama-3.2-1B; `run_g06.sh`, `src/69-71`) | `results/gens_summary_g06_all.txt`, `results/g06_*_selection.txt`, `results/llm_prompts_*.json` |
| G07 | sampling parameters (top_k 250, top_p 0.9, temperature 0.8 / 1.2 vs library default top_k 50, T=1); entropy matching (HW2): teacher-forced entropy of real vs generated accompaniment, then top_k 250 and T1.2/T1.4 + k250 configs | done (entropy measurement and two matched configs; results/entropy_match_run1.json, entropy_match_run2.json, sp_t*k250_*) | `src/68_sampling_params.py`, `results/gens_summary_g08_g02b_g05_g07.txt` |
| G08 | era-contrast (delta) CFG: logits = L(neutral) + g (L(era) - L(neutral)), melody in both branches | done (g=3,5; also + top_k 250 at g=3) | `src/60_delta_guidance.py`, `results/gens_summary_g08_g02b_g05_g07.txt` |
| G02b | CFG sweep (1, 3, 5, 7) with seeds 1 and 2 (the existing sweep is seed 0 only) | done | `results/gens_summary_g08_g02b_g05_g07.txt` |
| G08b | guidance sweep on MuseControlLite; negative-prompt guidance on MusicGen (opposite-era contrast, g=3, g=5, + top_k 250); APG not applicable to the discrete-token MusicGen path (APG is a diffusion-sampler technique) and not done for MuseControlLite | MuseControlLite sweep done; negative prompt g=3 and g=3+k250 done, g=5 running | `src/60b_negprompt_guidance.py`, `results/neg_g3*_*.json` |
| G09 | best-of-N sampling with an independent selector (LALM judge or CLAP) plus identity constraint (HW3) | done (8 candidates, seeds 10-17, 3 eras) | `src/67_bestofn_analysis.py`, `results/bon_summary.txt` |
| G10 | accompaniment level in the remix (-12..+6 dB) | done | `remixlvl_*.json` |
| G11 | LoRA on era accompaniment: v1 (r16, 3000), v2 (r16, 1000), v3 (r4, 3000) | done | `lora*_eval.json` |
| G11b | LoRA training-seed replicates v1s1, v1s2 | done | `lora_v1s{1,2}_*`, `gens_summary_lora_seeds.txt` |
| G11c | LoRA target modules (attention only vs + feed-forward), lr, text dropout, training on mix chroma vs none | lr (3e-5, 1e-4, 3e-4; seed 0, plus seed 1 for 3e-5 and 3e-4), target modules (attention vs + feed-forward) and prompt dropout (0, 0.1, 0.3) done, one training seed each; chroma-conditioning ablation (text-only LoRA, `run_lora_nomel.sh`) running | `run_lora_lr.sh`, `results/gens_summary_cfg3s3_lora_lr.txt`, `results/gens_summary_lora_abl.txt` |
| G12 | MuseControlLite (Stable Audio Open + melody adapter) | done | `mcl_*.json` |
| G13 | ACE-Step 1.5 cover mode, strengths .2/.5/.8/1.0 | done | `ace_c*_*.json`, VAE control `ace_vae_id_*.json` |
| G14 | market-targeted covers on dataset B | done | `market_*.json` |
| G15 | other ACE-Step tasks / checkpoints (repaint, lego, base/sft) | todo, low priority | - |
### C. Evaluation validity
| ID | Question | Status | Evidence |
|---|---|---|---|
| E01 | codec-matched probes (EnCodec 2.2 kbps real accompaniment) | done | `probe_codec_*.json` |
| E02 | negative controls (white/pink noise, DSP baselines) | done | `controls_*.json` |
| E03 | identity rows (EnCodec round trip, ACE VAE round trip) | done | `gen_eval_codec_24s3.json`, `ace_vae_id_eval.json` |
| E04 | independent evaluators: CLAP zero-shot era, Discogs-EffNet tagger, Fx-Encoder++ | done for base / MCL; run for LoRA, ACE | `clap_era*.json`, `discogs_eval.json`, `fx_encoder.json` |
| E05 | distribution / quality: KAD-style margin, Audiobox, COCOLA | done for base, MCL, cfg, text-only; run for LoRA, ACE | `kad_*.json`, `quality_*.json` |
| E06 | Fx-Encoder++ production metric | done (base) | `fx_encoder.json` |
| E07 | HW3 metrics FD-VGG, KL-PaSST, CLAP-music checkpoint (av-benchmark) against per-decade real references | todo (check install; PA3 session is resolving the sources) | - |
| E08 | LALM judge (Qwen2-Audio-7B-Instruct, MQAScore-style): positive control on 132 real val clips, then generators | done (weak judge: real-clip accuracy near chance, Spearman .45-.67) | `lalm_era.json`, `lalm_summary.txt` |
| E09 | identity metrics: chroma-DTW song-ID, chord agreement, tempo | done | `content_*.json` |
| E10 | basic-pitch lead-line metrics | done | `basicpitch.json` |
| E11 | Discogs-VINet song-identity | done | `vinet_identity.json` |
| E12 | human listening | skipped by decision | limitation |
### D. Reproducibility (reviewers and the course)
| ID | Item | Status |
|---|---|---|
| R01 | `repro/environment.txt`: package versions per venv, HF revision hashes, third-party repo commits, manifest and checkpoint hashes | done 2026-10-28 |
| R02 | run registry: script -> command -> output (this file + WORKLOG headings; shell drivers `src/run_*.sh`) | partly; to complete with a single `repro/RUNS.md` generated from the shell scripts |
| R03 | determinism check: re-embed a sample twice in separate processes and compare; seeds fixed (`torch.manual_seed`) for generation; list non-deterministic parts (GPU kernels) | done |
| R04 | clean-environment regeneration of the main tables from `results/*.json` on the Mac (no GPU) via one script | todo |
| R05 | data and licence statement: PA1 audio and Demucs stems not published; only features and generated accompaniment demos | done (CLAUDE.md, Pages) |

## 4. Ranked next steps (one at a time)
1. Finish running jobs: cue retrain (A14), LALM judge (E08), independent evaluators for LoRA/ACE (E04/E05), delta guidance (G08), LoRA seeds (G11b).
2. Component-justification ablations that reviewers ask first: A17 (pooling), A18/A19 (probe regularization and learning curve), A23/A24 (language proxy validity).
3. Generation controls: G05, G06, G07, G09 (selector from E08), G11c.
4. Evaluation alignment with HW3: E07 (FD-VGG, KL-PaSST, CLAP-music) if install succeeds.
5. Reproducibility closure: R02-R04.
6. Last: report and presentation (not started by decision).

## 5. Known limitations to state in the paper
Validation-only reporting on small splits (A 132, B 102); no artist or album grouping available; language and genre are proxies; probes are trained on codec-resynthesized real audio and may still shift on other decoders (ACE VAE round trip measured, other generators not); generation evaluated on 24 songs; no human perceptual test; LoRA rank/steps differences are single-run unless G11b shows small seed variance; third-party survey claims marked "recalled" in `SURVEY3.md` are not verified.

