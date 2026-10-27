# Survey 3 synthesis (prompt v3, gap-focused), 2026-10-27

Raw replies: `survey3/claude.md`, `survey3/gemini.md`, `survey3/chatgpt.md`. Rule: only items re-checked against the source in this session count as verified; everything else is a lead to be checked before citation.

## 1. How far to trust each reply
| File | Verdict |
|---|---|
| `claude.md` | Substantive and specific, labels items "verified" vs "recalled". Spot checks this session all matched (below). Absence claims ("no work found") are limited to its search. |
| `gemini.md` | Lead list only. States as fact things the Discogs-VI page does not enumerate (per-version release country/date/master id fields) and that audio can be legally rebuilt in a week; its "novelty check" table lists our own project as rows; several citations point to ResearchGate/unrelated pages (for example the MDL-probing row cites arXiv 2312.10019, not Voita and Titov). Not cited. |
| `chatgpt.md` | Generic deep-research workflow template in Japanese; does not answer the numbered items. Not used. |

## 2. Verified this session (source opened)
- Lee, Lee, Ju, Kim, Lee, Jeong, "Measuring Cross-Cultural Style Diffusion Through Era Classification: US and Korean Popular Music", ISMIR 2026, arXiv 2608.10980 (submitted 2026-08-11). Billboard Hot 100 first-entry era from audio (22,002 tracks with audio), Melon about 2,200 tracks; artist-aware split by collaboration-graph components, 8:1:1; mono 16 kHz, random 30 s crops; six decade classes, macro accuracy 67.0 to 71.2% across architectures (18 runs: 69.0 +- 2.0), Melon median offset about -4.7 y (1960s) to -2.7 y (2000s). Section 7 limitations: models cannot separate compositional style from recording technology; 30 s crops lack song form; artist-aware split does not control producers/engineers/songwriters; future work: identify the acoustic cues behind the offset. Remaster check: 243 original/remaster pairs, changed predictions split about evenly earlier/later. Uses CNNs from scratch and avoids pretrained encoders (MERT, Jukebox) because of possible cultural leakage. Stem separation: not stated.
- Hsieh, Cheng, Fan, Yang, Yang, "Addressing the confounds of accompaniments in singer identification", arXiv 2002.06817: separates vocals with open-unmix and proposes "shuffle-and-remix" of vocals with accompaniments of other songs.
- Belrose et al., LEACE, arXiv 2306.03819: closed-form linear concept erasure.
- Liang, "A Diagnostic Evaluation Framework for AI-Generated Cover Songs...", arXiv 2607.19688: 30 covers, 5 songs, 6 systems, expert severity ratings; harmony 53% and arrangement 47% severe-error rates; no feature correlation survives multiplicity correction. (Annotator count not stated on the abstract page.)
- SongEcho, arXiv 2602.19976 (ICLR 2026 per arXiv comments): conditions on the original vocal melody and text; generates new vocals and accompaniment together.
- Discogs-VI dataset page (mtg.github.io/discogs-vi-dataset): about 1.9 M versions in about 348 k cliques; Discogs-VI-YT about 493 k versions in about 98 k cliques; metadata CC BY-NC-SA 4.0, code AGPLv3; no audio shared, YouTube IDs plus download scripts; features on request for non-commercial research; per-version field list NOT enumerated on the page.

## 3. What this changes in our positioning (interpretation)
1. Closest prior work is Lee et al. 2026. It names the composition-vs-production confound as its main limitation and did no stem analysis, which is what our stem and production controls address. Cite it as the closest work, not claim priority over it.
2. Not found by the reply (not verified by us): any audio study predicting release country/market; geographic-origin work labels by artist residence on world music (UCI set; verified by the reply only). Phrase as "to our knowledge" and do not claim more.
3. Not found by the reply: era as an evaluated control in melody-preserving cover generation. Same wording rule.

## 4. Gaps in our own work that the replies point to (and what we do about each)
| Item (source) | Status | Plan |
|---|---|---|
| Shuffle-and-remix to see which stem a probe follows (Hsieh 2020, verified) | not done | E1: recombine vocals of clip i with accompaniment of clip j (different label), loudness-matched, embed with MERT, apply the mix probes, report which label the prediction follows |
| LEACE concept erasure of language, with matched random-erasure control (Belrose 2023 verified; caveat paper 2307.15054 recalled) | not done | E2: erase the Whisper-LID language direction from MERT embeddings, re-probe market/era, compare to erasing random directions of the same rank |
| Instrumental / low-vocal strata, language strata | partly (gap_A/B) | E4: probe accuracy by vocal-share tercile and by LID group |
| Release country vs artist origin divergence rate | not done | E5: descriptive statistic on Discogs-VI metadata, only if the downloaded files contain release country and artist fields (not confirmed) |
| Remaster/reissue check (Lee 2026) | not possible with PA1 (no Discogs ids) | state as limitation |
| External era benchmark (Billboard-Melon release) | needs YouTube audio | not run; terms of use unclear; ask course staff |
| Discogs-VINet as identity metric for covers (reply, recalled) | not done | check installability, then add next to chroma-DTW |
| Human listening test (replies, Liang 2026) | built, no participants | parked |

## 5. Unverified items we must not cite yet
Everything marked "recalled" in `claude.md`; MPEcho, MuLaCover, Qwen-Music numbers; ACE-Step 1.5; Amazon singing-language-ID paper; ICCS 2024 geographic origin paper; Fraunhofer artist-filter paper; all Gemini-only citations.
