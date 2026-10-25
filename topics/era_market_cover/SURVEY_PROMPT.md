# Literature survey prompt: era and release-market cues, and cover-song generation

Paste everything below the line into a deep-research tool. Save the reply as `SURVEY.md` in this folder.

---

You are helping plan a graduate course final project (Deep Learning for Music Analysis and Generation, NTU, music generation track). I need a literature survey that I can act on within a very short time window. Please be rigorous: every paper, dataset and system you cite must be real and carry a link (arXiv id, DOI or project page). If you cannot verify something, say "unverified" instead of guessing. Do not invent numbers. Separate what a source reports from your own interpretation.

## The project idea (offered by the course TA as an example topic)
Extend a prior homework (HW1: classify the release era and the release market of a song from audio) in two steps:
1. Using a larger dataset (Discogs-VI was suggested), compare how the original mix, the isolated vocals and the isolated accompaniment each contribute to era and market classification, and analyze which eras or markets are most confused. The goal is to tell whether a model learns singing language, arrangement and recording style, or dataset bias. Note that the market label is where a song was released, not necessarily where the artist is from.
2. If stable era-style cues exist, try generating covers of the same song in different era styles, and check that the melody is kept while the target era's characteristics appear.

## My background (for feasibility judgments)
From HW1 I have working code for frozen-embedding probing of pretrained audio encoders (MERT, MuQ, CLaMP3, MAEST, Whisper and others) with logistic regression or SVM heads, layer selection, leakage-free out-of-fold fusion, and zero-shot audio-language-model prompting with calibration. I care about honest evaluation: confidence intervals and paired significance tests rather than point estimates. I have no prior hands-on experience with generative audio or source separation.

## Constraints
- A team of a few students, found through a course forum. Time is short: roughly 3 to 4 weeks for the whole project, and my own GPU access ends within about a week, so plan for teammates' compute or small experiments. Audio data may be large; prefer designs that use precomputed embeddings.
- The project must fit the music generation track, so a generation component should exist, but a solid analysis half is acceptable if the generation half is small and honestly evaluated.

## What I need (please answer each numbered item)

1. **Closest prior work, verified.** For each of: (a) music era / decade / year prediction from audio, (b) release-region, country or market prediction from audio, (c) stem-wise or source-separated analysis of what a music classifier uses (vocals vs accompaniment), (d) dataset bias and shortcut learning in music tagging or classification (artist effects, album effects, recording or mastering artifacts, language), list the 5 to 8 most relevant papers or systems with year, venue, what they did, headline result as reported, and what they did not do. Say which ones are the real competitors for this project and which are only background.
2. **Datasets.** Discogs-VI and any alternative with era and region metadata (for example Discogs-based sets, Million Song Dataset derivatives, MTG-Jamendo, FMA, Free Music Archive, MusicBrainz or AcousticBrainz based sets, any set with release country). For each: size, labels, how audio is obtained (full audio, previews, features only), license and redistribution limits, known leakage issues (artist or album overlap between splits), and whether it is realistic to use in under two weeks. Recommend one primary and one fallback dataset.
3. **Source separation as an analysis tool.** Which separators are standard and usable (for example Demucs variants, Spleeter, BS-RoFormer or MDX-style models)? Cost per song, known artifacts, and the risk that separation artifacts leak era or format information (for example mono versus stereo, bandwidth). Any prior work that used separated stems to analyze a classifier.
4. **Era and market cues, what is known.** What does the literature say about acoustic correlates of era (production and mastering, loudness trends, instrumentation, harmony, language of vocals) and of market (language, scales, instruments)? Which are likely shortcuts rather than musical style? Suggest control experiments that separate language from music (for example instrumental only subsets, language identification as a covariate).
5. **Cover-song or style-transfer generation.** Survey systems that can generate a version of a given song in another style while keeping the melody: melody-conditioned text-to-music (for example MusicGen-Melody and similar), audio-to-audio style transfer, diffusion and ControlNet-style conditioning (Music ControlNet, MuseControlLite and others), stem-based or accompaniment generation, singing-voice conversion, any "cover generation" or "song-to-song restyle" work. For each: inputs, what is kept versus changed, open-source availability, compute needed, and quality caveats. Mark what could run on a single GPU within days.
6. **Evaluation of generated covers.** How should melody preservation and era-style presence be measured? Review options such as chroma or CQT similarity with alignment, melody extraction agreement, cover-song-identification embeddings, a frozen era classifier as an evaluator (and the circularity risk if the same classifier guided generation), CLAP score, Frechet audio distance, and listening tests with few participants. Recommend a small, defensible protocol with paired statistics.
7. **Novelty check.** Has anyone studied era or market classification stem by stem, or controlled generation by era cues learned from a classifier? State what appears to be new and what is already done. Do not overclaim.
8. **Three concrete project variants**, each with: research question, data, models, experiments in order, what to cut if time runs out, expected deliverable, main risk, and a rough effort estimate in person-days. Variant A should be analysis-heavy with a small generation demo; variant B generation-heavy with a light analysis; variant C a different angle you think is more interesting and novel than the TA's example. Rank the three by feasibility and by interest, and say which you would pick.
9. **Reading list.** Ten papers or resources, ordered, with one line on why each matters, for a student who has one weekend to prepare.

## Output format
Plain structured sections numbered as above, tables where they help, every claim with its source link, a final section "Unverified or uncertain" listing anything you could not confirm, and no filler.
