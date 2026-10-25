# Survey synthesis: era and release-market cues, cover-song generation

Merges three deep-research replies to `CommE5070_FP_era_market_cover_survey_prompt.md` (raw files in `survey1/`). Checks marked "verified" were done on 2026-10-25 against the source page; everything else is as reported by the replies.

## 1. How far to trust each reply
| File | Verdict |
|---|---|
| `survey1/topic2_survey1_claude.md` | Most reliable and most specific (dataset fields, controls, systems, protocol). Spot checks pass: He et al. 2024 (arXiv:2407.05368, 54% within 3 years, audio only), Discogs-VI facts, SongEcho (ICLR 2026, code released). Many 2026 arXiv ids and the claims about them are not yet checked. |
| `survey1/topic2_survey1_deepresearch.md` (Japanese) | Sound general plan, but weak on fit: proposes FMA as primary data, which has no release-market or era labels; its closest-work table is mostly background (aesthetic-score shortcut paper is real, ISMIR 2026, but unrelated to era). Its Discogs-VI audio statements are not supported by the dataset page. |
| `survey1/topic2_survey1_gemini.md` | **Do not cite from it without checking.** "U-Mix" (Maman and Bermano, IEEE SPL 2022) was not found by search; "Kwon et al. arXiv:2108.01869" is actually an unrelated reinforcement-learning paper; says Discogs-VI audio comes "course-provided", which is unsupported. Some well known papers in it (Sturm 2014, Flexer 2007, MusicGen, HTDemucs, MERT) are real. |

## 2. What all three agree on (use as the design)
- Stems: separate once with HT Demucs (2-stem or 4-stem), cache, then embed with frozen encoders (MERT, MAEST, MuQ, CLaMP3, Whisper). CPU-only probing afterwards.
- Split by artist (group K-fold); report random-split inflation as a control (Flexer 2007).
- Shortcut controls: accompaniment-only for market; Whisper language ID on vocals as a covariate; loudness and bandwidth equalization (mono, LUFS normalization, common low-pass); recombined vocals + accompaniment as a separator control.
- Generation: MusicGen-Melody (chroma-conditioned, 1.5B) is the realistic open model; vocals are not generated, so the clean design is to keep the original vocal stem and regenerate the accompaniment.
- Evaluation: chroma/DTW or pitch-track agreement for melody; a frozen era probe from a different encoder family than the one used for analysis (circularity); CLAP and FAD as secondary; small 2AFC listening test.

## 3. Verified facts that change the plan
- Discogs-VI ships no audio (only YouTube ids and a download script); about 1.9M versions, 348k cliques; Discogs-VI-YT about 493k versions, 98k cliques; metadata CC BY-NC-SA 4.0; CQT features on request for non-commercial research. Source: https://mtg.github.io/discogs-vi-dataset/
- Year and country come from the Discogs release data and one version maps to many releases, so era and market labels need a rule (for example earliest release). The survey that says per-track year and country are absent from the main files (Claude reply) was not re-checked against the files themselves.
- Era recognition from audio exists (He et al., 2024, arXiv only, no stems, no market, no shortcut analysis). Release-market from audio: nothing close found.
- SongEcho (arXiv:2602.19976, ICLR 2026) shows cover generation is already an active area; do not claim novelty there.

## 4. Candidate scope (my recommendation)
1. Start from the HW1 data (already labeled and downloaded); add Discogs-VI-YT subset only if a teammate can download reliably.
2. Core (Variant A of the replies): mix vs vocals vs accompaniment vs recombined, frozen encoders, artist-group splits, paired tests, controls above. This is the part that can be finished on cached embeddings.
3. Added angle from the Claude reply (its Variant C): use Discogs-VI cliques (same composition, different era or market) as natural experiments, predicting which of two versions is older from embedding differences, per stem.
4. Generation: MusicGen-Melody on accompaniment only, 5 to 12 songs, 3 target eras, evaluated against real target-era versions where a clique has one.

## 5. Open questions for the team
- Which HW1 dataset and labels do we start from, and how many tracks can teammates process?
- Who has a GPU for separation and embedding extraction (a few days of one GPU is enough if clips are 30 s)?
- Do we keep the market label, or era only, if separation or download slows us?
