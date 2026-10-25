# SURVEY: era and release-market cues, stem-wise analysis, and era-style cover generation

Prepared 2026-10-25 for the CommE5070 final project (music generation track). Every source below was located and checked during this survey unless it is listed in the final section "Unverified or uncertain". "Reported" means the number comes from the cited source; "Interpretation" marks my own reading.

---

## 0. Five things that change the plan (read this first)

1. **Discogs-VI does not ship audio, and its main files do not carry year or country per track.** The dataset gives metadata plus YouTube IDs only; audio must be downloaded by you, and the authors note they got banned by YouTube when they used too many parallel download processes. CQT features exist but are "on request" for non-commercial research. Per-track records contain `release_id`, artist IDs and `formats` (Vinyl, CD, Cassette, Shellac...), not year or country; those must be joined from the Discogs release dump using `release_id`. Source: https://mtg.github.io/discogs-vi-dataset/ and the example clique https://mtg.github.io/discogs-vi-dataset/data/example_clique.json
2. **One "version" in Discogs-VI maps to many releases** (reissues, compilations, different countries, different years). So "release era" and "release market" of a recording are not single-valued. You must define a rule (for example earliest release year, and country of earliest release, or keep only versions whose releases agree). Interpretation: this is the single biggest label-noise risk for the project.
3. **Discogs-VI's cliques are a gift for the research question.** A clique is a set of versions of the same composition, often recorded decades apart and in different countries. Holding the composition fixed is the cleanest available control for "is the model hearing the song, or the era/production?" (see Variant C).
4. **Cover-song generation became an active topic in 2026** (SongEcho, MPEcho, Qwen-Music, MuLaCover). Do not claim "first cover generation". What I did not find is era-targeted cover generation evaluated with a stem-aware era probe and real same-song covers from the target era as a reference.
5. **MAEST was pretrained on ~3.3M tracks matched to Discogs metadata** (Discogs20, in-house MTG). Its embeddings were trained to predict Discogs styles, which correlate with era and region, and its training audio likely overlaps Discogs-VI recordings. Not label leakage in the strict sense, but report it. Source: https://huggingface.co/mtg-upf/discogs-maest-30s-pw-73e-ts

---

## 1. Closest prior work, verified

### (a) Era / decade / year prediction from audio

| Paper | Year, venue | What they did | Headline result (as reported) | What they did not do |
|---|---|---|---|---|
| He, Song, Hao, Wang, Lu, Li, "Music Era Recognition Using Supervised Contrastive Learning and Artist Information" [arXiv:2407.05368](https://arxiv.org/abs/2407.05368) | 2024, arXiv (ByteDance/Fudan) | Frames era recognition as classification; audio-only model with supervised contrastive learning, plus a multimodal variant using artist info | On MSD, audio-only model reaches 54% accuracy within a 3-year tolerance; adding artist info with their MMC framework gives about 9% more | No stem separation, no analysis of what cues are used, no market label, no shortcut controls |
| Bertin-Mahieux et al., Million Song Dataset / YearPredictionMSD [UCI 203](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd) | 2011 | Year regression benchmark from 90 Echo Nest timbre statistics, 515,345 songs, years 1922 to 2011 | Benchmark, no single headline; official split avoids the "producer effect" by keeping each artist in one split | Features only, no audio, so no stems; mostly Western |
| Mauch, MacCallum, Levy, Leroi, "The evolution of popular music: USA 1960–2010" [DOI 10.1098/rsos.150081](https://doi.org/10.1098/rsos.150081) | 2015, R. Soc. Open Sci. | ~17,000 Billboard Hot 100 recordings, harmonic and timbral topics over time | Continuous change with faster "revolutions" around 1964, 1983 and 1991 | Not a classifier study; US charts only |
| Serrà et al., "Measuring the evolution of contemporary western popular music" [arXiv:1205.5651](https://arxiv.org/abs/1205.5651) | 2012, Sci. Rep. | Pitch, timbre, loudness statistics over >50 years | Restricted pitch transitions, homogenized timbre, growing loudness | Not a classifier; MSD-derived descriptors |
| Deruty & Pachet, "The MIR perspective on the evolution of dynamics in mainstream music" [ISMIR 2015 PDF](https://archives.ismir.net/ismir2015/paper/000136.pdf) | 2015, ISMIR | Dynamics descriptors across years and 272 genres | Dynamics relate primarily to a track's year rather than genre; loudness war peaks around 2007 for their descriptors | Not a classifier; but directly tells you loudness is an era shortcut |
| Okada & Miura, decade estimation for Japanese female idol music [JASA abstract](https://labs.sonicfield.org/library/computational-estimation-of-released-decades-for-japanese-women-idol-music-from-) | 2016, JASA (abstract) | Logistic regression on 25 acoustic parameters, 10-fold CV | 58.56% accuracy, kappa 0.48 | Single genre/market; no deep models; split protocol unclear |

### (b) Release region / country / market prediction from audio

| Paper | Year, venue | What they did | Result | Not done |
|---|---|---|---|---|
| Zhou, Claire, King, "Predicting the Geographical Origin of Music" [DOI 10.1109/ICDM.2014.73](https://doi.org/10.1109/ICDM.2014.73), data at [UCI 315](https://archive.ics.uci.edu/dataset/315/geographical+original+of+music) | 2014, ICDM | Regression to latitude/longitude from MARSYAS features | Public set is 1,059 tracks, 33 countries; Western music deliberately excluded; origin = artist's main country of residence | Not release market; traditional music only; features only |
| "Artist filtering for non-western music classification" [Fraunhofer record](https://publica.fraunhofer.de/handle/publica/374409) | Fraunhofer conference paper (year: see Unverified) | Region-of-origin classification with and without artist filters, per feature domain | Studies how the artist effect inflates region classification | Not release market; pre-deep-learning |
| Choi & Wang, LRID-Net, "Listen, Read, and Identify" [arXiv:2103.01893](https://arxiv.org/abs/2103.01893) | 2021, ISMIR | Singing language ID from audio + metadata | Shows language is identifiable from audio; uses Music4All with artist-disjoint 80:20 split | Not market; relevant as the language control |

Interpretation: there is essentially no prior work on *release market* (where a release came out) from audio. Region-of-origin work is the nearest, and its main lesson is the artist effect.

### (c) Stem-wise / source-separated analysis of what a classifier uses

| Paper | Year, venue | What they did | Result | Not done |
|---|---|---|---|---|
| Haunschmid, Manilow, Widmer, "audioLIME" [arXiv:2008.00582](https://arxiv.org/abs/2008.00582), code [CPJKU/audioLIME](https://github.com/CPJKU/audioLIME) | 2020, workshop/arXiv | LIME where perturbations switch separated sources on/off (Spleeter), giving listenable explanations of taggers | Works on two taggers where a competing method fails; authors flag separator artifacts as a limitation | Tagging only; no era/market; no group-level stem ablation |
| Haunschmid et al., "Towards Musically Meaningful Explanations Using Source Separation" [arXiv:2009.02051](https://arxiv.org/abs/2009.02051) | 2020, arXiv | audioLIME on a classifier deliberately trained with a spurious confounding source | Detects the confound easily | Synthetic confound, not era |
| Garoufis et al., "Pre-training Music Classification Models via Music Source Separation" [arXiv:2310.15845](https://arxiv.org/abs/2310.15845) | 2023/24 | Separation objectives as pretraining for tagging/genre | Accompaniment-separation pretraining gives a statistically significant genre gain; vocal pretraining hurts genre | Not an attribution analysis |
| "The Vocal Eras Tour: Microtiming Trends Across Decades and Genres" [TISMIR 10.5334/tismir.278](https://transactions.ismir.net/articles/10.5334/tismir.278) | 2025/26, TISMIR | Microtiming of source-separated vocal stems across seven decades | Decade/genre trends in vocal timing; separated-stem noise required manual checks | Descriptive, not classification |
| Ensemble genre classification on separated stems [arXiv:2412.15602](https://arxiv.org/abs/2412.15602) | 2024, arXiv | Separate into 2/4/5 stems, classify each, fuse | Stem fusion for genre | Genre only, no shortcut analysis |

### (d) Dataset bias and shortcut learning in music classification

| Paper | Year, venue | Point |
|---|---|---|
| Sturm, "A simple method to determine if a MIR system is a 'horse'" [DOI 10.1109/TMM.2014.2330697](https://doi.org/10.1109/TMM.2014.2330697) | 2014, IEEE TMM | Controlled interventions show three state-of-the-art systems get high scores without musical knowledge; small tempo changes can collapse them |
| Rodríguez-Algarra, Sturm, Dixon, "Characterising Confounding Effects in Music Classification Experiments through Interventions" [DOI 10.5334/tismir.24](https://doi.org/10.5334/tismir.24) | 2019, TISMIR | Procedure of pipeline interventions plus resampling to characterise confounding (demonstrated on GTZAN). Directly reusable as your methodology frame |
| Flexer & Schnitzer, "Effects of Album and Artist Filters in Audio Similarity Computed for Very Large Music Databases" [Project MUSE](https://muse.jhu.edu/article/393608) | 2010, Computer Music Journal | Artist effect persists in very large databases and the album effect is larger than the artist effect; cites a drop from 71% to 27% when an artist filter is applied in earlier work |
| Flexer, "A closer look on artist filters for musical genre classification" [ISMIR 2007 PDF](https://archives.ismir.net/ismir2007/paper/000341.pdf) | 2007, ISMIR | Artist filters lower accuracy and can erase differences between methods |
| Sturm & Flexer, "Validity in Music Information Research Experiments" [arXiv:2301.01578](https://arxiv.org/abs/2301.01578) | 2023, arXiv | Review of validity threats; good framing for the report |

**Real competitors vs background.** Competitors (you must cite and position against): He et al. 2024 (era recognition), audioLIME (stem-based attribution), Rodríguez-Algarra et al. 2019 (intervention methodology), and, for the generation half, SongEcho / MuseControlLite / MusicGen-Melody (Section 5). Everything else is background.

---

## 2. Datasets

| Dataset | Size | Labels relevant here | Audio | License / redistribution | Leakage issues | Usable in < 2 weeks? |
|---|---|---|---|---|---|---|
| **Discogs-VI / Discogs-VI-YT** ([site](https://mtg.github.io/discogs-vi-dataset/), [Zenodo 13983028](https://doi.org/10.5281/zenodo.13983028), [arXiv:2410.17400](https://arxiv.org/abs/2410.17400)) | ~1.9M versions / ~348k cliques; YT subset ~493k versions / ~98k cliques | Clique (composition), artist IDs, release IDs, physical `formats`. Year and country via join to Discogs release dump (`discogs_20240701_releases.xml.jsonl` in the 8.7 GB intermediary zip) | YouTube IDs only; you download | Metadata CC BY-NC-SA 4.0; code AGPLv3; audio not shared; CQT features on request | Same recording appears in many releases (grouped as a version); artists recur across cliques; official splits are designed for VI, not for era/market | Only a **subset** (a few thousand 30 s clips) and only if a teammate can download reliably. Budget 2 to 4 days for download + join + cleaning |
| **DiVers** ([arXiv:2608.04543](https://arxiv.org/abs/2608.04543)) | >1.1M versions | Splits compatible with Discogs-VI-YT; automatic tags such as instrumental and live | Metadata (audio presumably via links; see Unverified) | See paper | Includes amateur/live content, which is a different production domain | Useful only for its `instrumental` tag as a language control |
| **YearPredictionMSD** ([UCI 203](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)) | 515,345 | Year 1922 to 2011 | None (90-dim timbre stats) | UCI | Artist-disjoint official split | Yes, but no stems possible; only a sanity baseline |
| **FMA** ([arXiv:1612.01840](https://arxiv.org/abs/1612.01840), [repo](https://github.com/mdeff/fma)) | 106,574 tracks, 16,341 artists, 14,854 albums | Genre, artist, album metadata; dates and artist location fields (see Unverified) | Full audio, CC licensed; 30 s clips subset ~93 GiB | Creative Commons per track | Album/artist effects; web-era independent music, so era range is narrow | Yes for audio, but a weak fit for era and market |
| **Music4All** (used in [arXiv:2103.01893](https://arxiv.org/abs/2103.01893)) | 109,269 30 s clips | Language labels (46 languages, estimated from lyrics with langdetect; heavily English), lyrics, metadata | 30 s clips | Access by agreement (see Unverified) | No official split; artist overlap must be handled | Possible if access is quick; good for the language control |
| **Geographical Origin of Music** ([UCI 315](https://archive.ics.uci.edu/dataset/315/geographical+original+of+music)) | 1,059 tracks | Lat/long of origin | Features only | UCI | Tiny | Background only |
| **AcousticBrainz-based sets** | — | — | Features only; project discontinued (see Unverified) | — | — | Not recommended |
| **Your HW1 dataset** | (you know it) | Era + market already labeled | You already have audio | Course | Already handled leakage in HW1 | Yes, immediately |

**Recommendation.** Primary: a **Discogs-VI-YT subset** of roughly 3k to 8k versions, sampled to balance decades (e.g. 1960s to 2010s) and 6 to 8 release countries, restricted to cliques with ≥2 versions in different decades (needed for Variant C), audio cut to 30 s from the track middle. Fallback: **your HW1 dataset** (already labeled, already downloaded), re-run with stems. Do not wait on Discogs-VI downloads before starting the stem pipeline; develop it on HW1 audio in week 1.

Label rules to fix in advance (interpretation): era = decade of the *earliest* release of that version; market = country of that earliest release; drop versions whose releases span >1 decade before the reissue boom, or report both "strict" and "loose" label sets. Split by **artist** (group K-fold on `release_artist_ids`), and report a clique-disjoint split as a secondary check.

---

## 3. Source separation as an analysis tool

| Separator | Stems | Reported quality | Availability | Notes |
|---|---|---|---|---|
| HT Demucs (Rouard, Massa, Défossez) [arXiv:2211.08553](https://arxiv.org/abs/2211.08553), [code](https://github.com/facebookresearch/demucs) | 4 (vocals, drums, bass, other); 2-stem mode | 9.20 dB SDR on MUSDB with extra data (sparse fine-tuned variant reported higher) | Open, pip-installable | De facto default. Works on CPU or GPU |
| BS-RoFormer (Lu, Wang, Kong, Hung) [arXiv:2309.02612](https://arxiv.org/abs/2309.02612) | 4 | 1st place in SDX'23 music separation track; 9.80 dB average SDR on MUSDB18HQ without extra data (smaller model) | Official weights not confirmed; community checkpoints exist (see Unverified) | Best vocals quality in practice; Mel-RoFormer is a follow-up |
| Spleeter (Hennequin et al. 2020) | 2/4/5 | Older | Open | Used by audioLIME; faster, more artifacts |
| Open-Unmix | 4 | Older baseline | Open | Not recommended for this project |

**Cost per song.** I could not verify a reliable seconds-per-song figure; measure on 10 tracks of your 30 s clips before committing. Precompute stems once, then embed; never separate inside a training loop.

**Known artifacts and leakage risks.**
- Separated stems contain bleed and artifacts; audioLIME's authors flag this explicitly, and the Vocal Eras Tour study had to filter noisy onsets in separated vocals. Perceptual quality metrics (SDR, FAD) correlate poorly with listener ratings on vocal stems ([Bake-Off, arXiv:2507.06917](https://arxiv.org/abs/2507.06917)).
- Interpretation: separation quality itself is era-dependent. Mono 1950s/60s masters, narrow bandwidth, tape hiss and lossy YouTube re-encodes will separate differently from modern stereo masters. A probe on "vocals" can therefore learn era from *separation residue*, not from singing.
- Controls I recommend (my design, not from a paper):
  1. **Recombined control:** classify `vocals + accompaniment` re-summed from the separator. Mix vs recombined isolates "separator processing" effects.
  2. **Format equalization:** downmix to mono, low-pass all audio to a common bandwidth (e.g. the lowest found in old material), and loudness-normalize (LUFS) before embedding. Report accuracy before and after.
  3. **Format probe:** predict the Discogs `formats` field (Shellac/Vinyl/Cassette/CD/File) and simple signal stats (bandwidth, stereo width, integrated loudness, crest factor) from the same embeddings; if these predict era nearly as well as the encoder, you have a shortcut.

---

## 4. Era and market cues: what is known

**Era (reported).** Increasing loudness and decreasing crest factor through the loudness-war years, with dynamics tied more to year than to genre (Deruty & Pachet 2015). Timbral homogenization and restricted pitch-transition repertoire over time (Serrà et al. 2012). Harmonic/timbral topic shifts with rapid change around 1964, 1983 and 1991 in US charts (Mauch et al. 2015). Vocal microtiming differs across decades (Vocal Eras Tour).

**Market (interpretation, weakly supported).** Language of vocals is the obvious cue (LRID-Net shows language is audibly identifiable). Scales, instruments and local genres matter for traditional music (Zhou et al. 2014), but for commercial pop the release-country label is dominated by language, label/distribution practices, and artist identity. Note that market ≠ origin: an English-language US hit reissued in Japan is "JP market".

**Likely shortcuts rather than style.** Loudness/mastering level, bandwidth and noise floor of the source (format, transfer, YouTube upload quality), mono vs stereo, artist and album identity (Flexer), language for market, and Discogs metadata artefacts such as which releases got matched to YouTube at all.

**Control experiments.**
- Language: run Whisper language ID on the separated vocal stem (a pipeline like this, Demucs vocals then an LID model, is used in [StepAudio 3 Music report, arXiv:2609.16034](https://arxiv.org/abs/2609.16034)); then (i) report market accuracy within single-language subsets, (ii) add predicted language as a covariate and test whether audio embeddings add information beyond it (likelihood-ratio or nested-model comparison), (iii) test on an instrumental-only subset (DiVers `instrumental` tag or a vocal-energy threshold such as the 0.08 RMS ratio used in the Vocal Eras Tour).
- Artist: group splits by artist; compare to random split to quantify the artist effect on your data.
- Production: the three separator/format controls in Section 3.
- Composition: within-clique evaluation (Variant C).

---

## 5. Cover-song / style-transfer generation

| System | Inputs | Kept vs changed | Open? | Compute | Caveats | Runs on 1 GPU in days? |
|---|---|---|---|---|---|---|
| **MusicGen-Melody** (Copet et al., NeurIPS 2023) [arXiv:2306.05284](https://arxiv.org/abs/2306.05284), [audiocraft](https://github.com/facebookresearch/audiocraft) | Text + chromagram of reference audio | Keeps harmonic/melodic chroma; changes arrangement, timbre, production | Yes, 1.5B "melody" checkpoint | Inference only; README says a GPU is required | Chroma conditioning blurs octave/voicing; instrumental only (no intelligible vocals); 32 kHz | **Yes** |
| **MusicGen-Style** (Rouard et al., ISMIR 2024) [arXiv:2407.12563](https://arxiv.org/abs/2407.12563), [model card](https://huggingface.co/facebook/musicgen-style) | Text + 1.5 to 4.5 s audio style excerpt | Imitates the style of the excerpt; no melody lock | Yes | Inference only | Can copy/loop the conditioning if too much info passes; text may be ignored when audio conditioning is strong (reviewer notes) | **Yes**; useful to feed a real target-era excerpt as the style reference |
| **MuseControlLite** (Tsai, ..., Yang; ICML 2025) [arXiv:2506.18729](https://arxiv.org/abs/2506.18729), [project](https://musecontrollite.github.io/web/) | Text + time-varying melody/rhythm/dynamics + optional reference audio, on Stable Audio Open | Melody (and optionally rhythm/dynamics) kept; style from text | Code + checkpoints released | 85M trainable params for fine-tuning; inference on SAO | Trained on MTG-Jamendo only; slightly worse KL/CLAP than SAO ControlNet; instrumental | **Yes** (inference). Note: from NTU (Yi-Hsuan Yang's group) |
| **Stable Audio Open ControlNet** (Hou et al., ICASSP 2025) [arXiv:2410.05151](https://arxiv.org/abs/2410.05151) | Text + melody on Stable Audio DiT | Melody kept, text-driven style transfer explicitly evaluated | Demo page; code status unverified | — | Instrumental | Unverified |
| **Music ControlNet** (Wu, Donahue, Watanabe, Bryan; 2023/24, journal venue unverified) [arXiv:2311.07069](https://arxiv.org/abs/2311.07069) | Text + melody, dynamics, rhythm | Reported 49% more faithful to input melodies than MusicGen with 35x fewer parameters | Weights not found (Adobe) | — | Mel-spectrogram U-Net; 2023-era quality | No |
| **Stable Audio Open** (Evans et al. 2024) [arXiv:2407.14358](https://arxiv.org/abs/2407.14358) | Text (and audio-to-audio via init audio in stable-audio-tools) | Text-to-audio up to 47 s stereo 44.1 kHz | Open weights, CC-trained | Inference only | Weaker on music than prior Stable Audio; CC training data skews away from commercial pop eras | Yes, as a base |
| **ACE-Step / ACE-Step 1.5** [arXiv:2506.00045](https://arxiv.org/abs/2506.00045), [v1.5 report arXiv:2602.00744](https://arxiv.org/abs/2602.00744), [repo](https://github.com/ace-step/ACE-Step-1.5) | Text tags + lyrics + source audio | v1 "only_lyrics" mode keeps original melody while changing tags; v1.5 lists cover generation, repainting, vocal-to-BGM | Open (MIT for v1.5 weights per model card) | Consumer-GPU claims in report (unverified for your GPU) | Full songs with vocals; melody preservation not guaranteed or measured by you yet | **Probably yes**; test early |
| **SongEcho** (Li et al., ICLR 2026) [arXiv:2602.19976](https://arxiv.org/abs/2602.19976), [code](https://github.com/lsfhuihuiff/SongEcho_ICLR2026) | Vocal pitch (RMVPE) + text tags + lyrics, on ACE-Step | Vocal melody kept; new vocals and accompaniment generated in text style | Code, dataset (Suno70k) and demos released | Trained on 3 A100s for 30k steps; inference cost unverified | English-only training data from AI-generated songs; style via free tags | Inference only, if checkpoints are released (check) |
| **MPEcho, Qwen-Music, MuLaCover** [2607.26698](https://arxiv.org/abs/2607.26698), [2607.11699](https://arxiv.org/abs/2607.11699), [2610.26684](https://arxiv.org/abs/2610.26684) | Melody (+ phonemes / lead sheet) + lyrics + style text | Melody kept, style rewritten | Mostly unverified | Large | Very recent | No (background) |
| **SingSong** (Donahue et al. 2023) [arXiv:2301.12662](https://arxiv.org/abs/2301.12662) | Isolated vocals → accompaniment | Keeps the vocal exactly; generates accompaniment | Not released | — | Listeners preferred SingSong over a retrieval baseline | No, but its **design is the one to copy** |
| **JASCO** (Tal et al., ISMIR 2024) [arXiv:2406.10970](https://arxiv.org/abs/2406.10970) | Text + chords, melody, drums | Local symbolic/audio controls | In audiocraft (see Unverified) | — | Instrumental | Possibly |
| Singing-voice conversion (RVC, so-vits-svc) | Vocal stem | Changes singer timbre only | Community tools, no canonical paper | Light | Ethics/consent issues; not an era effect | Not recommended |

**Interpretation: the cleanest small-scale design.** Separate the original into vocals + accompaniment; keep the original vocal stem untouched; regenerate only the accompaniment in a target-era style (MusicGen-Melody conditioned on the original accompaniment's chroma, or MuseControlLite with the vocal melody as condition and an era text prompt); remix with the original vocals. Melody is preserved by construction, so evaluation can focus on whether era cues appear in the accompaniment. This also ties directly to the stem analysis: if the accompaniment probe carries era information, this is exactly where generation should act.

---

## 6. Evaluating generated covers

**Melody preservation.**
- Pitch-track agreement: extract vocal or lead melody with RMVPE ([arXiv:2306.15412](https://arxiv.org/abs/2306.15412)) and score with `mir_eval` Raw Pitch Accuracy, Raw Chroma Accuracy, Overall Accuracy. This is the protocol used by SongEcho.
- Chroma cosine similarity on aligned frames (introduced as the melody metric in MusicGen). Add DTW alignment if tempo is allowed to change.
- Version-identification embedding distance with CLEWS (Serrà, Araz, Bogdanov, Mitsufuji; ICML 2025) [arXiv:2502.16936](https://arxiv.org/abs/2502.16936), [code](https://github.com/sony/clews), checkpoints trained on Discogs-VI exist. Calibrate it: compute the distribution of CLEWS distances between real versions in the same clique and between random pairs; report where generated covers fall. This gives an interpretable "as similar as a real cover" scale.

**Era-style presence.**
- A frozen era probe as evaluator. **Circularity:** if any probe guided prompt choice or sampling, evaluate with a probe built on a *different* encoder family (e.g. guide with MERT, evaluate with MuQ or CLaMP3) and trained on a disjoint artist set. Even then, the probe may reward shortcuts (loudness, bandwidth); run it on format-equalized audio (Section 3).
- Real-reference direction: for songs whose clique contains a real version from the target era, measure whether the generated cover's embedding moves toward that real version relative to the original (cosine to target minus cosine to source). Interpretation: this is less circular than a classifier score.
- CLAP score ([arXiv:2211.06687](https://arxiv.org/abs/2211.06687), LAION-CLAP) between audio and "a 1970s recording" style prompts: cheap, but text-era alignment in CLAP is unvalidated; report as secondary.
- FAD per era with fadtk ([arXiv:2311.01616](https://arxiv.org/abs/2311.01616), [code](https://github.com/microsoft/fadtk)): FAD of generated-era-X against real-era-X accompaniment stems vs against other eras. Use FAD∞ or the per-song mode; FAD is biased at small sample sizes, which is your regime.

**Listening test with few participants.** Interpretation: a 2AFC design is the most defensible at n≈10 to 15. For each item, play original + two covers (target era vs a different era, same seed family), ask "which sounds more like the 1970s" and "does it keep the tune" (yes/no). Analyze with a mixed-effects logistic regression (random effects for listener and song) or, simpler, an exact binomial test per condition plus a paired sign test across songs. Pre-register the number of songs (e.g. 12) and the hypotheses.

**Recommended protocol (small, defensible).**
1. 12 to 20 source songs from held-out artists; 3 target eras; 3 seeds each.
2. Primary metrics: RPA on the vocal (should be ≈ original by construction if vocals are kept), CLEWS calibrated distance, held-out-encoder era probe accuracy on format-equalized audio.
3. Paired statistics per song: Wilcoxon signed-rank or paired bootstrap (resample songs, not clips) for "target-era probability after vs before"; Holm correction across eras.
4. Baselines: (i) original accompaniment re-remixed (identity), (ii) generated accompaniment with a neutral prompt ("a recording"), (iii) a simple DSP "era filter" (band-limit + mono + reduced loudness) to show whether the probe is fooled by production alone. Baseline (iii) is important: if it scores as "older" as well as the generator does, the probe is reading production artifacts.

---

## 7. Novelty check

What appears already done:
- Era recognition from audio (He et al. 2024, older decade studies).
- Region-of-origin prediction and artist-effect analysis for it (Zhou et al. 2014; Fraunhofer artist filtering).
- Source separation for explanations of taggers (audioLIME) and for stem-wise genre classification (several smaller works).
- Melody-preserving style transfer and cover-song generation (MusicGen-Melody, SAO ControlNet style transfer, MuseControlLite, SongEcho, MPEcho, Qwen-Music, MuLaCover).

What I did not find in these searches (so it may be new, but treat as "not found", not "proven absent"):
- Era **and release-market** classification analysed stem by stem with explicit separator, format and language controls.
- Using **cover cliques as natural experiments** to hold composition fixed while varying era/market for a classifier.
- Era-targeted cover generation evaluated with a stem-aware era probe **plus** real target-era covers of the same song as references, with a DSP production-only baseline.

Do not claim novelty for "release market as a label" in general; the HW1 task presumably already defines it. Do not claim novelty for cover generation.

---

## 8. Three project variants

### Variant A. Analysis-heavy: "Where do era and market live? A stem-wise, controlled probe study" (+ small demo)

- **RQ:** How much era/market information is in vocals vs accompaniment vs production/format, after controlling for artist, language and separator artifacts?
- **Data:** HW1 data first; Discogs-VI-YT subset if downloads succeed.
- **Models:** HT Demucs (2-stem); frozen MERT, MuQ, CLaMP3, MAEST, Whisper encoder; your LR/SVM probes with OOF fusion.
- **Experiments in order:** (1) separate + embed mix, vocals, accompaniment, recombined; (2) probes per input, group-split by artist, CIs by bootstrap over artists, McNemar between inputs on the same items; (3) format-equalized rerun; (4) format and signal-statistics probes; (5) language covariate via Whisper LID; (6) confusion analysis by era pairs and market pairs; (7) demo: MusicGen-Melody accompaniment regeneration for 5 songs × 3 eras.
- **Cut if late:** step 4 and the demo's listening test.
- **Deliverable:** table of accuracy by input × encoder with CIs; control deltas; confusion matrices; 15 demo clips.
- **Main risk:** weak generation component for a generation-track course.
- **Effort:** ~22 to 28 person-days.

### Variant B. Generation-heavy: "Era-targeted accompaniment regeneration with melody kept by construction"

- **RQ:** Can off-the-shelf melody-conditioned generators make an existing song sound like a target era while keeping its melody, and does a probe agree with listeners?
- **Data:** 20 held-out songs (vocals + accompaniment); reference real-era accompaniment stems for FAD and probe training.
- **Models:** HT Demucs; MusicGen-Melody and MusicGen-Style (style excerpt from a real target-era track); optionally MuseControlLite or ACE-Step cover mode; RMVPE; CLEWS; one era probe (light version of Variant A).
- **Experiments:** (1) pipeline on 3 songs end to end; (2) prompt/guidance sweep on a dev set; (3) full generation 20 × 3 eras × 3 seeds; (4) objective metrics; (5) 2AFC listening test (n≈12); (6) DSP-only baseline.
- **Cut if late:** MuseControlLite/ACE-Step comparison; keep MusicGen only.
- **Deliverable:** demo page, metric table with paired tests, listening-test results.
- **Main risk:** MusicGen outputs may sound era-ambiguous or low quality at 32 kHz; probe/listener disagreement.
- **Effort:** ~25 to 32 person-days.

### Variant C (my pick). "Covers as natural experiments: what does an era/market model hear when the song is held fixed, and can we move a song across eras?"

- **RQ1:** Within a Discogs-VI clique (same composition), can a probe still tell era and market apart, and from which stem? This removes composition and, often, lyrics as confounds.
- **RQ2:** For a held-out song, does accompaniment regeneration toward era X move its embedding toward the *real* era-X version of the same song?
- **Data:** Discogs-VI-YT cliques with ≥2 versions from different decades (and, for market, different countries), ~1k to 2k cliques, 30 s clips; year/country joined from the release dump with the earliest-release rule.
- **Models:** as Variant A for analysis; as Variant B for generation; CLEWS for alignment of segments and melody similarity (the new Discogs-VI-SIREN similar-regions release on the dataset site can help pick comparable segments, see Unverified).
- **Experiments:** (1) data join + label rules (2 days); (2) standard era/market probes, artist-split; (3) **pairwise within-clique task**: given two versions of the same song, predict which is older (LR on embedding differences), per stem and per format-equalized input; (4) the same for "same vs different market"; (5) language covariate; (6) generation: 12 to 20 songs with a real version in the target era; directional metric toward the real target-era cover, RPA, held-out-encoder probe, DSP baseline; (7) small 2AFC test.
- **Cut if late:** market half of (4); listening test reduced to the authors plus a few classmates and reported as pilot.
- **Deliverable:** within-clique stem attribution results (the novel analysis), generated covers with a reference-anchored evaluation.
- **Main risk:** YouTube download yield and label noise from reissues; mitigate by building on HW1 data first and adding cliques as they arrive.
- **Effort:** ~28 to 35 person-days (fits 4 people × 3 weeks if the generation half stays inference-only).

**Ranking.** Feasibility: A > B > C. Interest/novelty: C > B > A. **I would pick C**, scoped so that its first two analysis steps are exactly Variant A on HW1 data (no wasted work if Discogs downloads fail), and its generation half is Variant B with MusicGen-Melody only.

**Compute note (interpretation).** Everything except generation is CPU-friendly once stems and embeddings are cached; do separation and embedding extraction on your GPU *this week*, store per-layer pooled embeddings (float16) and stems at 16 to 24 kHz mono if your encoders accept it. Generation (MusicGen 1.5B inference) can run on a teammate's single GPU.

---

## 9. Weekend reading list (in order)

1. Sturm 2014, "horse" paper [DOI](https://doi.org/10.1109/TMM.2014.2330697): the mindset for the whole analysis half.
2. Rodríguez-Algarra, Sturm, Dixon 2019 [TISMIR](https://doi.org/10.5334/tismir.24): intervention procedure you can copy for controls.
3. Flexer & Schnitzer 2010 [CMJ](https://muse.jhu.edu/article/393608): artist and album effects at scale.
4. Araz, Serra, Bogdanov 2024, Discogs-VI [arXiv:2410.17400](https://arxiv.org/abs/2410.17400) + the dataset site: how versions, cliques and releases relate.
5. He et al. 2024, music era recognition [arXiv:2407.05368](https://arxiv.org/abs/2407.05368): the direct competitor.
6. Haunschmid et al. 2020, audioLIME [arXiv:2008.00582](https://arxiv.org/abs/2008.00582): stems as explanation units, and their artifact caveat.
7. Deruty & Pachet 2015 [ISMIR](https://archives.ismir.net/ismir2015/paper/000136.pdf): why loudness is an era shortcut.
8. Copet et al. 2023, MusicGen [arXiv:2306.05284](https://arxiv.org/abs/2306.05284): melody conditioning and the chroma metric you will use.
9. Tsai et al. 2025, MuseControlLite [arXiv:2506.18729](https://arxiv.org/abs/2506.18729) (skim SongEcho [arXiv:2602.19976](https://arxiv.org/abs/2602.19976) for the RPA/RCA protocol).
10. Gui et al. 2024, Adapting FAD [arXiv:2311.01616](https://arxiv.org/abs/2311.01616): why small-sample FAD is biased and how to use fadtk.

---

## Unverified or uncertain

- Separation runtime per song for HT Demucs / BS-RoFormer on your hardware: not verified; measure.
- Official BS-RoFormer weights: not found; community checkpoints exist (e.g. one used in a CPJKU restoration paper, [arXiv:2603.04032](https://arxiv.org/abs/2603.04032)) but licensing was not checked.
- Whether the Discogs release dump fields `released` and `country` are present in `discogs_20240701_releases.xml.jsonl` exactly under those names: the dataset site states Discogs dumps contain release year and country, but I did not open that file.
- Discogs-VI-SIREN (similar regions between versions, computed with CLEWS): announced on the dataset site with a Zenodo DOI; contents not inspected.
- DiVers audio availability and license: only the abstract was checked.
- FMA field names for release date and artist location: believed present in `tracks.csv` / artist metadata but not verified in this survey.
- Music4All access terms (agreement-based) and whether it includes release year: not verified.
- AcousticBrainz being discontinued: from memory, not re-verified here.
- Year and venue of "Artist filtering for non-western music classification" (Fraunhofer): authors and year not confirmed.
- MusicGen-Melody / MusicGen-Style weight licenses (believed non-commercial): not re-verified; fine for coursework but check before publishing audio.
- Music ControlNet weights: none found; Stable Audio Open ControlNet (Hou et al.) code release not confirmed.
- SongEcho trained checkpoints: the paper says code, dataset and demos are released; whether inference-ready weights are included was not checked.
- JASCO availability in audiocraft: not re-verified.
- ACE-Step consumer-GPU claims and "only_lyrics keeps the melody": taken from project pages/tutorial descriptions, not tested.
- Vocal Eras Tour exact publication year and authors: page located, details not fully read.
- He et al. 2024 venue beyond arXiv: not confirmed.
