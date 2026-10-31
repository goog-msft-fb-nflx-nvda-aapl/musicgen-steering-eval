# CommE5070 final project (music generation track)

Course project for CommE5070 (Deep Learning for Music Analysis and Generation, NTU). The repository holds two independent topics, each in its own folder with its own README, code, results, work log and plan. People interested in either topic are welcome to join.

| Topic | Folder | One line |
|---|---|---|
| 1. Steering evaluation | [`topics/musicgen_steering/`](topics/musicgen_steering/) | Independent, significance-tested evaluation of concept-direction steering in MusicGen (four direction methods, independent evaluators, layer selection). |
| 2. Era and market cues, era-targeted covers | [`topics/era_market_cover/`](topics/era_market_cover/) | Stem-wise analysis of release-decade and release-market cues on Discogs-VI clips, and cover generation that keeps the original vocal and regenerates the accompaniment in a target decade. |

## Team
- James Christian: MS in AI, UT Austin
- Angel Hsia: MS, [Graduate Institute of Communication Engineering (GICE), NTU](https://gice.ntu.edu.tw/en/)
- Name to be announced: Ph.D., GICE, NTU

## Shared
- `docs/` GitHub Pages source (kept at the repository root because Pages serves from `/docs`): `index.html` for topic 1, `era_market/` for topic 2. Pages URL: https://goog-msft-fb-nflx-nvda-aapl.github.io/musicgen-steering-eval/
- Topic 1 mid-progress video: https://youtu.be/aaZ89FoZe_U
- `LICENSE` Apache License 2.0.

Each topic folder is self-contained: run its scripts from inside that folder.

## License
Code and text in this repository are released under the Apache License 2.0 (`LICENSE`). Datasets, audio and third-party models used by the experiments keep their own licenses and terms (for example the Discogs-VI metadata CC BY-NC-SA 4.0, course-provided audio, and model weights from their authors); none of that audio is redistributed here except the generated demo clips under `docs/era_market/` and `docs/audio/`, which are model outputs released under the same Apache-2.0 license as the code (the licenses of the generators, for example MusicGen-Melody CC-BY-NC 4.0 weights, still apply to their use). Code under AGPL (Discogs-VINet) is used as an external checkout and is not included.
