# musicgen-steering-eval

Does concept steering in MusicGen survive a fair evaluation? We add a learned direction inside a frozen MusicGen so the output is pulled toward a target musical note, then test it with paired, significance-tested evaluation, four direction methods, and layer selection experiments. Course project for CommE5070 (Deep Learning for Music Analysis and Generation, NTU), music generation track. Status: mid-progress. People interested in the topic are welcome to join.

- Interactive page (video, listen to steered vs unsteered clips, headline numbers): https://goog-msft-fb-nflx-nvda-aapl.github.io/musicgen-steering-eval/
- Mid-progress video: https://youtu.be/aaZ89FoZe_U
- Every run with its full config and result: `WORKLOG.md`. Design: `PLAN.md`. Open items: `TODO.md`. Literature notes: `SURVEY.md`.

## Questions
1. Does steering hold up under evaluators independent of the steering probe?
2. Which direction method (RFM, mean-difference, logistic regression, LDA) steers best at matched audio fidelity?
3. Are the most decodable layers also the layers where steering works best?

## Layout
- `src/` pipeline: `01` train note directions, `03` steer and compare arms, `04` validate chroma evaluator, `05` cache hidden states, `06` other direction methods, `07` CLAP fidelity, `08` controls, `09` layer sweep, `10a/10b` MERT second evaluator, `11` layer predictors, `analysis_*.py` statistics (CPU only, run on the stored results).
- `results/` per-clip results and summaries; `artifacts/` trained directions and probes (see `artifacts/README.md`).
- `docs/` the interactive page and the speaker script of the video.

## Reproduce the analysis (no GPU)
```
pip install numpy scipy
python src/analysis_methods.py small      # method comparison, paired McNemar
python src/analysis_ablation.py           # medium high eta, rfm components, training seeds
python src/analysis_confusion.py results/ablation/arms_eval2b
```
Generation and training need a GPU, `transformers`, the MusicRFM code base (github.com/astradzhao/music-rfm) and the SynTheory dataset; see the headers of the scripts in `src/` for options.

## Caveats
Automatic evaluators (chroma, a MERT note probe, CLAP) are not a listening test, and no listening test has been run yet. One concept family (root notes) and one model family so far.

## License
Code and text in this repository are released under the Apache License 2.0 (`LICENSE`). Datasets, audio and third-party models used by the experiments keep their own licenses and terms (for example the Discogs-VI metadata CC BY-NC-SA 4.0, course-provided audio, and model weights from their authors); none of that audio is redistributed here except the generated demo clips under `topics/era_market_cover/`. Code under AGPL (Discogs-VINet) is used as an external checkout and is not included.
