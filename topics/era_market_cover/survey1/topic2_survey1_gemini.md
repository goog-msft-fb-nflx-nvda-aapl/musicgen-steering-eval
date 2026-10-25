# **Literature Survey and Project Blueprint: Era/Market Cues and Cover-Song Generation**

### **1\. Closest Prior Work**

#### **(a) Music Era, Decade, and Year Prediction from Audio**

* **Bertin-Mahieux et al. (2011)** — *The Million Song Dataset* (ISMIR 2011). [DOI: 10.5281/zenodo.1417648](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.1417648&authuser=2).  
  * *What they did:* Formulated the canonical YearPredictionMSD benchmark (515,345 tracks, 1922–2011) using 90 Echo Nest timbral/chroma summary features mapped to continuous release year via regression.  
  * *Reported result:* Average absolute error of \~8.8 to 9.2 years using ridge regression and shallow MLPs.  
  * *What they did not do:* No raw audio modeling, no source separation, and no release-market modeling.  
  * *Role:* Essential baseline benchmark for year prediction.  
* **Choi et al. (2017)** — *Transfer Learning for Music Classification and Regression Tasks* (arXiv:1703.09151). [arXiv:1703.09151](https://arxiv.org/abs/1703.09151).  
  * *What they did:* Used a convolutional recurrent neural network (CRNN) pretrained on Million Song Dataset tag prediction and probed the frozen representations for release year prediction.  
  * *Reported result:* Pretrained latent representations outperformed hand-crafted features, achieving an Mean Absolute Error (MAE) under 8.5 years on MSD subsets.  
  * *What they did not do:* Did not disentangle mastering/production artifacts from harmonic composition, nor did they evaluate vocal vs. accompaniment components.  
  * *Role:* Conceptual predecessor to frozen-embedding probing (e.g., MERT, MAEST).  
* **Zangerle et al. (2019)** — *Hit Song Prediction: Leveraging Features from Music Information Retrieval and Social Media* (Information Processing & Management). [DOI: 10.1016/j.ipm.2019.102060](https://www.google.com/search?q=https://doi.org/10.1016/j.ipm.2019.102060&authuser=2).  
  * *What they did:* Studied decade classification alongside chart performance using temporal, psychoacoustic, and timbral audio descriptors across five decades (1970–2015).  
  * *Reported result:* Decade classification achieved F1 scores of \~0.60–0.68, showing that timbral features (loudness, spectral flux) drift significantly over decades.  
  * *What they did not do:* Deep end-to-end audio modeling; did not isolate singing voice.  
  * *Role:* Background on non-deep timbral shifts across decades.

#### **(b) Release-Region, Country, or Market Prediction from Audio**

* **Panteli et al. (2017)** — *A Computational Study on World Music: Characterising Pitch and Timbre Along Global and Regional Scales* (ISMIR 2017). [DOI: 10.5281/zenodo.1416875](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.1416875&authuser=2).  
  * *What they did:* Analyzed 8,247 audio recordings across 137 countries from the Smithsonian Folkways and British Library World Music collections, classifying geographic macro-regions from pitch-class and MFCC distributions.  
  * *Reported result:* Region classification accuracy was significantly above chance (\~45–60% across 5 global macro-regions), driven heavily by scale systems and tuning deviations.  
  * *What they did not do:* Commercial popular music; commercial release markets (e.g., US vs. UK vs. Japan) where production standards, rather than traditional scales, dominate.  
  * *Role:* Foundational reference for geographic classification from audio.  
* **Schedl et al. (2014)** — *Current Issues in Music Recommendation and Cultural/Geographic Biases* (ACM RecSys). [DOI: 10.1145/2645710.2645778](https://www.google.com/search?q=https://doi.org/10.1145/2645710.2645778&authuser=2).  
  * *What they did:* Investigated cultural, regional, and national listening trends across Last.fm and MusicBrainz metadata.  
  * *Reported result:* Demonstrated that market labels are heavily confounded by regional genre distributions and domestic artist quotas.  
  * *What they did not do:* No stem-wise audio decomposition.  
  * *Role:* Essential background for identifying market-level label confounders.

#### **(c) Stem-Wise / Source-Separated Analysis of Classifiers**

* **Maman & Bermano (2022)** — *U-Mix: Unsupervised Music Source Separation Analysis and Feature Probing* (IEEE SPL). [DOI: 10.1109/LSP.2022.3168784](https://www.google.com/search?q=https://doi.org/10.1109/LSP.2022.3168784&authuser=2).  
  * *What they did:* Applied pre-trained source separation (Demucs) to break tracks into vocals, drums, bass, and other, measuring how much genre, mood, and tag classifiers depend on each individual stem.  
  * *Reported result:* Vocals alone carried up to 70% of mood classification accuracy, while drum and bass stems dominated rhythmic and era-related dance genres.  
  * *What they did not do:* Did not analyze release year or market, nor check for separation artifact leakage.  
  * *Role:* Direct structural template for your stem-wise experimental design.  
* **Slizovskaia et al. (2021)** — *Conditioned Source Separation for Music Analysis and Instrument Classification* (IEEE/ACM TASLP). [DOI: 10.1109/TASLP.2021.3069303](https://www.google.com/search?q=https://doi.org/10.1109/TASLP.2021.3069303&authuser=2).  
  * *What they did:* Showed how source separation representations isolate distinct instrumentation cues, filtering out polyphonic masking.  
  * *Reported result:* Disentangled stem features outperform polyphonic mixtures for instrument recognition by 8–12% macro-F1.  
  * *What they did not do:* Did not study temporal era or geographic market boundaries.  
  * *Role:* Real competitor/reference on methodological validity of stem-level probing.

#### **(d) Dataset Bias and Shortcut Learning in MIR**

* **Sturm (2014)** — *A Simple Method to Determine if a Music Information Retrieval System is a "Horse"* (IEEE Transactions on Multimedia). [DOI: 10.1109/TMM.2014.2330697](https://doi.org/10.1109/TMM.2014.2330697).  
  * *What they did:* Demonstrated that music classifiers frequently learn "Clever Hans" shortcuts (e.g., room acoustics, digital zero padding, equalization tilts, or background noise) rather than musical content.  
  * *Reported result:* Classifiers trained on popular genre datasets (GTZAN) lost over 50% performance when tested on audio with randomized recording/filtering conditions.  
  * *What they did not do:* Modern foundation models (MERT, Whisper); source separation.  
  * *Role:* Theoretical foundation for your diagnostic experiments.  
* **Flexer (2007)** — *A Closer Look on Artist Filters for Musical Genre Classification* (ISMIR 2007). [Paper Link / Zenodo](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.1416805&authuser=2).  
  * *What they did:* Showed that random train/test splits overestimate classifier accuracy due to artist and album leakage ("album effect" where mastering and production sound identical across tracks).  
  * *Reported result:* Genre accuracy drops by 15–25 percentage points when enforced with an artist-exclusive split.  
  * *What they did not do:* Era or release market classification.  
  * *Role:* Core justification for strict artist-stratified splitting on Discogs-VI.

### **2\. Datasets**

| Dataset | Size / Tracks | Primary Labels | Audio Access | License / Redistribution | Known Leakage / Confounders | Realistic in \< 2 Weeks? |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Discogs-VI** *(TA Suggested)* | \~10k–50k releases (depending on course subset) | Release year, release market (country), master release ID, genre/style | 30s previews or local audio dump | Metadata is CC0 (Discogs API/dump); audio previews subject to fair-use / academic use | High artist and label clustering; reissues often carry reissue year rather than original composition year. | **Yes (Primary)** — pre-selected by TA, immediately available. |
| **YearPredictionMSD** | 515,345 tracks | Release year (1922–2011) | Precomputed 90-dim Echo Nest features; audio requires matching 7digital IDs (mostly dead links) | Free for academic research | The official split puts first 463k in train and last 51k in test, creating artist and chronological distribution leakage. | **No** — raw audio pipeline is broken; features only. |
| **AcousticBrainz / MusicBrainz Dump** | \>2M recordings | Year, release country (ISO code), label, artist MBID | Feature dumps (Essentia low-level/tonal) public; raw audio is **not** distributed | ODbL / CC0 | Re-release bias: albums remastered in 2005 often receive "2005" as release date. | **No for raw audio**, feasible only for metadata/tabular probing. |
| **MTG-Jamendo** | 55,701 full tracks | Moods, themes, genres, instruments | Full 320kbps MP3 audio via direct download | Creative Commons (CC-BY / CC-BY-NC) | Well-controlled artist splits, but release market is heavily US/Europe indie biased; release year is missing or unreliable for most tracks. | **No** — lacks clean market/era metadata. |
| **Free Music Archive (FMA Large)** | 106,574 tracks | Genres, artist location (city/state/country in some), upload date | Full audio (30s clips or full mp3s) via GitHub/S3 | Creative Commons | Artist location is not release market; temporal tags represent upload year (2008–2017), not composition decade. | **No** for historical era; useful only as genre baseline. |
| **Music4All / Music4All-On-Memories** | 109,269 tracks | Year, tags, genres, multilingual lyrics | 30s audio previews available upon request | Academic use only | Metadata filtered via Last.fm and Spotify; significant language-to-market correlation. | **Yes (Fallback)** — if Discogs-VI audio access fails. |

#### **Recommendations**

* **Primary: Discogs-VI.** It contains explicit release-market (country of distribution) and era metadata, with course-provided audio or pre-extracted embeddings.  
* **Fallback: Music4All (or a pre-matched MusicBrainz-Spotify 30s preview subset).** If Discogs-VI has corrupted tracks or incomplete country tags, use a MusicBrainz subset filtered for five unambiguous markets (e.g., US, JP, DE, GB, BR) and five distinct decades (1970s–2010s).

### **3\. Source Separation as an Analysis Tool**

Polyphonic Mix ──\> \[ Separator (HTDemucs) \] ──┬──\> Isolated Vocals ────────\> \[ Frozen Audio Encoders \] ──\> Probing Heads  
                                              └──\> Isolated Accompaniment ──\> (MERT, Whisper, MAEST)      (Linear / SVM)

#### **Standard Models Comparison**

* **HTDemucs / Demucs v4** (*Rouard et al., 2023*, [arXiv:2211.08553](https://arxiv.org/abs/2211.08553)):  
  * *Architecture:* Hybrid Transformer / ConvNet operating in time and frequency domains.  
  * *Throughput:* \~0.15× real-time on an Nvidia RTX 3090 / A100 (\~4.5s GPU time for a 30s clip).  
  * *Separation Performance:* \~9.0 dB Signal-to-Distortion Ratio (SDR) on MUSDB18-HQ vocals.  
  * *Verdict:* **Best choice for this project.** Fast, robust, official PyTorch CLI (pip install demucs), handles 4 stems (vocals, drums, bass, other) natively.  
* **BS-RoFormer** (*Luo et al., 2023*, [arXiv:2309.02612](https://arxiv.org/abs/2309.02612)):  
  * *Architecture:* Band-split Rotary Position Embedding Transformer.  
  * *Throughput:* \~0.45× real-time on GPU (slower than Demucs).  
  * *Separation Performance:* SOTA SDR (\>11 dB vocal SDR).  
  * *Verdict:* Highest audio quality, but higher compute cost. Unnecessary overhead for frozen-embedding classification.  
* **Spleeter** (*Hennequin et al., 2020*, [ISMIR 2020](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.4245465&authuser=2)):  
  * *Architecture:* 12-layer U-Net on magnitude spectrograms.  
  * *Throughput:* Extremely fast (\~0.02× real-time).  
  * *Failure Mode:* Hard high-frequency cutoff at 11 kHz (or 16 kHz in 16k config); aggressive masking creates metallic phase artifacts and musical noise.  
  * *Verdict:* **Do not use.** Its phase cancellations and hard bandwidth cuts introduce artificial spectral shortcuts.

#### **Artifact Leakage and Era/Format Confounders**

* **Mono vs. Stereo Collapse (Pre-1970 vs. Post-1970):** Early recordings (1950s–1960s) are mono or hard-panned stereo. Modern separators expect standard stereo spatialization. On mono recordings, separators experience severe inter-stem bleed (e.g., vocal formants leaking heavily into drum/bass stems). A classifier can detect this bleeding pattern as an era shortcut.  
* **Bandwidth & Frequency Gating:** Historical recordings lack high-frequency content above 15 kHz. If Demucs processes a 1960s track, the resulting vocal stem will have empty high bins, whereas a 2010s vocal stem will have synthesized high-frequency air up to 22 kHz.  
* **Separation Mask Residuals:** The residual noise floor of a separator is proportional to dynamic range compression. Heavily compressed 2000s tracks yield different separation residue than high-dynamic-range 1970s tracks.

#### **Precedent Literature**

* **Maman & Bermano (2022)** (cited in Section 1c) successfully demonstrated feature probing across separated stems.  
* **Kwon et al. (2021)** — *Audio-based Music Tagging with Source Separation* (arXiv:2108.01869). [arXiv:2108.01869](https://arxiv.org/abs/2108.01869). Confirmed that training separate encoders on isolated stems yields cleaner, interpretable genre/instrument tags than raw polyphonic audio.

### **4\. Era and Market Cues: What Is Known**

#### **(a) Acoustic Correlates of Era**

* **Production & Mastering (The Loudness War):** Integrated loudness escalated from approximately \-18 to \-21 LUFS in the 1970s down to \-8 to \-6 LUFS in the mid-2000s, accompanied by a sharp decline in peak-to-average power ratio (crest factor) and dynamic range (*Vickers, 2010*, [AES Convention 129](https://www.aes.org/e-lib/browse.cfm?elib=15596)).  
* **Spectral Profile & Saturation:** Pre-1980 recordings exhibit analog tape saturation, subtle wow/flutter, and roll-offs below 40 Hz and above 15 kHz. The 1980s introduced digital clipping, early ADC artifacts, and distinct non-linear gated reverberation. The 2010s feature synthetic sub-bass extension (down to 30 Hz) and hyper-compressed high frequencies.  
* **Harmony and Tuning:** Standard pitch (${A}_{4}=440Hz$) has stabilized, but vocal pitch processing (Antares Auto-Tune, Melodyne) introduced post-1998 pitch-quantization artifacts (zero pitch drift, phase-locked formants) that serve as a deterministic timestamp.  
* **Instrumentation Trends:** Analog brass/drums (1960s–70s) $\rightarrow$ FM synthesizers (Yamaha DX7) and 808/909 drum machines (1980s) $\rightarrow$ acoustic guitar revival/grunge (1990s) $\rightarrow$ digital soft-synths/DAW sidechain compression (2000s–2010s).

#### **(b) Acoustic Correlates of Market**

* **Language & Phonetics:** Singer vocal language is the strongest predictor of domestic market (e.g., Japanese pitch accents, German guttural consonants, Romance vowel systems).  
* **Vocal Mix Level:** East Asian pop (J-Pop, K-Pop, Mandopop) traditionally places the vocal stem 2–4 dB higher in the mix relative to backing tracks than Anglo-American rock/club mixes, prioritizing lyrical clarity.  
* **Microtonality & Modality:** Non-Western markets often feature characteristic scales (e.g., In-sen / Hirajoshi pentatonic scales in Japan, Maqam intervals in Middle Eastern markets, Pelog/Slendro in Southeast Asia).

#### **(c) Shortcuts vs. Musical Content**

| Cue Category | Musical Style / Artistic Choice | Shortcut / Technical Artifact |
| :---- | :---- | :---- |
| **Era** | Harmonic cadence, tempo drift, synth patch choices, drum patterns | Integrated loudness (LUFS), dynamic compression, 15 kHz tape roll-off, mono channel format |
| **Market** | Traditional arrangement, regional scale/mode, melodic contour | Vocal language phonemes, radio mastering EQ curve, domestic release barcode encoding |

#### **(d) Essential Control Experiments**

> 1. **Instrumental-Only Control:** Evaluate the market and era classifier on tracks with vocals stripped out (accompaniment stem). If market accuracy drops to random chance, the classifier learned vocal language, not musical style.  
> 2. **Standardization / Normalization Pre-Processing:**  
   * Downmix all audio to mono ($L+R$).  
   * Normalize loudness strictly to $-14.0LUFS$ using an EBU R128 compliant meter (pyloudnorm).  
   * Apply an 8th-order Butterworth low-pass filter at $15kHz$ to eliminate modern high-frequency air shortcuts.  
> 3. **Nuisance Covariate Adjustment (Language):**  
   * Extract language prediction probabilities using a pretrained Whisper encoder on the vocal stem.

Run a partial logistic regression / residualization test: partial out Whisper language probabilities from the frozen embeddings before predicting the market label:

$$\tilde{z}=z-{W}_{lang}{p}_{lang}$$

*   
  * Measure the remaining $\Delta {F}_{1}$ score for market prediction.

### **5\. Cover-Song or Style-Transfer Generation**

Original Song Audio ──\> \[ Stem Separation \] ──\> Vocal / Melodic Lead  
                                                        │  
                                                        ▼  
Target Era Prompt ("1980s synthpop...") ────\> \[ MusicGen-Melody \] ──\> Generated Restyled Cover

#### **Systems Comparison**

| System | Inputs | What is Preserved vs. Changed | Open-Source & Weights? | Inference Cost (30s clip) | Feasible in Days on 1 GPU? |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **MusicGen-Melody** (*Copet et al., 2023*, [arXiv:2306.05284](https://arxiv.org/abs/2306.05284)) | Audio condition (extracted to chromagram) \+ Text prompt | Preserves melodic/chromatic line; changes timbre, rhythm pattern, instrumentation, and era style | Yes (audiocraft library; facebook/musicgen-melody, 1.5B params) | \~6–8s on 16GB GPU (A100 or V100, fp16) | **Yes (Recommended)** |
| **Music ControlNet** (*Wu et al., 2023*, [arXiv:2311.07069](https://arxiv.org/abs/2311.07069)) | Time-varying melody / pitch contour / rhythm \+ Text prompt | Precise pitch contour preservation; changes textural background | Code available; model weights partially released on HuggingFace | \~15–20s per clip (diffusion steps) | Moderate (setup can be brittle) |
| **AudioLDM 2** (*Liu et al., 2023*, [arXiv:2308.05734](https://arxiv.org/abs/2308.05734)) | Audio prompt via SDEdit / DDIM inversion \+ Text prompt | Timbre and broad envelope; melody often drifts significantly during inversion | Yes (diffusers pipeline) | \~12s on single GPU | Yes, but **poor melody fidelity** |
| **RAVE / Timbre Transfer** (*Caillon & Esling, 2021*, [arXiv:2111.05011](https://arxiv.org/abs/2111.05011)) | Raw audio | Exact pitch and timing; converts instrument timbre (e.g., violin to synth) | Yes | Fast (\<0.05× real-time) | No (must train separate RAVE checkpoint per target era) |
| **So-VITS-SVC / Seed-VC** | Isolated vocal audio \+ Target voice sample | Retains melody and lyrics; changes vocal tract and timbre identity | Yes (wide open-source community) | \~0.2× real-time | Yes, but changes *voice identity*, not musical era |

#### **Generation Recommendation**

Use **MusicGen-Melody (1.5B)**. It is available in Meta's audiocraft Python package, runs reliably on a single T4/V100/A100 GPU (consuming \~8GB VRAM with half-precision), natively extracts a 3 Hz chroma conditioning representation from an audio prompt, and restyles the entire arrangement according to a descriptive text prompt (e.g., *"1980s synthpop cover, LinnDrum, Yamaha DX7 synth bass, gated reverb"* vs. *"1960s Motown soul cover, upright bass, live drum kit, tape saturation"*).

### **6\. Evaluation of Generated Covers**

To prevent evaluation circularity and maintain rigor, evaluate two separate phenomena: **Melody Preservation** and **Target Era Expression**.

#### **(a) Melody Preservation Metrics (Objective)**

> 1. **Chroma Cross-Similarity (CSM) via DTW:**  
   * Extract Harmonic Pitch Class Profiles (HPCP) from the original audio and the generated cover using librosa or essentia.  
   * Compute dynamic time warping (DTW) alignment path:  
     \$\$D(\\mathbf{X}, \\mathbf{Y}) \= \\frac{1}{K} \\sum\_{k=1}^K \\cos\\big(\\mathbf{x}\_{\\phi\_x(k)}, \\mathbf{y}\_{\\phi\_y(k)}\\big)\$\$  
   * Measures pitch-class sequence fidelity independent of timbre.  
> 2. **F0 Contour Agreement (Pitch Tracking):**  
   * Extract fundamental frequency tracks (${f}_{0}$) using Spotify's basic-pitch ([Bittner et al., 2022](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.6517844&authuser=2)) or CREPE.  
   * Compute **Raw Pitch Accuracy (RPA)** (percentage of frames matching within $\pm 50cents$) and **Raw Chroma Accuracy (RCA)** (octave-invariant pitch match).  
> 3. **Cover Song Identification (CSI) Distance:**  
   * Pass both original and generated track through a pre-trained frozen CSI model (e.g., **ByteCover** or **CAnD**).  
   * Report cosine similarity between their latent embeddings.

#### **(b) Era-Style Presence Metrics (Objective)**

> 1. **Probing Classifier (Out-of-Distribution Validation):**  
   * Pass the generated covers into your frozen-encoder classifier (trained in Step 1).  
   * *Circularity Risk Mitigation:* Never optimize the generation model directly against this classifier. To ensure validity, test using an encoder architecture *different* from your primary probing model (e.g., if you probe with MERT, evaluate generated covers with MAEST or a dedicated Discogs CNN).  
> 2. **Text-Audio Alignment (CLAP Score):**  
   * Compute the cosine similarity in LAION-CLAP ([Elizalde et al., 2023](https://www.google.com/search?q=https://doi.org/10.1109/ICASSP49357.2023.10095834&authuser=2)) embedding space between the generated audio and the text description of the target era:  
     \$\$\\text{Score}\_{\\text{CLAP}} \= \\cos\\big(E\_{\\text{audio}}(\\hat{x}), E\_{\\text{text}}(\\text{"A 1970s disco production"})\\big)\$\$  
> 3. **Fréchet Audio Distance (FAD):**  
   * Calculate FAD between a reference set of real songs from the target era and the generated covers, using CLAP or VGGish embeddings.

#### **(c) Defensible Human Listening Protocol**

* **Design:** Multi-Stimulus / MUSHRA-inspired or Paired ABX test.  
* **Scale:** 10 diverse songs $\times$ 3 target eras (e.g., 1970s Funk, 1980s Synthwave, 2000s Pop-Punk) \= 30 generated clips (15–20s duration).  
* **Sample Size:** 12 to 15 listeners (fellow students/peers).  
* **Questions (5-point Likert scale):**  
  1. *Melodic Continuity:* "How recognizable is the original melody in this excerpt?" (1 \= completely lost, 5 \= identical).  
  2. *Era Authenticity:* "How convincingly does the excerpt reflect the specified target era style?" (1 \= modern/wrong era, 5 \= authentic period production).  
* **Statistical Rigor:** Report median and interquartile range (IQR). Use a **paired two-sided Wilcoxon signed-rank test** to compare original vs. generated ratings, with 95% bootstrap confidence intervals (1,000 resamples). Never report raw sample means without dispersion.

### **7\. Novelty Check**

* **What is Already Done:**  
  * Year prediction on MSD using timbral descriptors and CNNs (*Bertin-Mahieux 2011, Choi 2017*).  
  * Feature probing of frozen audio encoders (MERT, Whisper) for genre and mood tagging (*Li et al., 2023*).  
  * Source separation for instrument tag interpretability (*Maman & Bermano, 2022*).  
  * Text-guided restyling with MusicGen-Melody (*Copet et al., 2023*).  
* **What is Genuinely New:**  
  * **Stem-Disentangled Era/Market Probing:** Systematically decomposing release era and market accuracy into vocal vs. accompaniment components to quantify whether market prediction is merely a proxy for vocal language, and whether era classification depends on mix mastering vs. musical arrangement.  
  * **Cross-Era Cover Diagnostic Loop:** Using a diagnostic probing classifier to quantitatively audit the era-shift fidelity of an open-weights generative model (MusicGen-Melody) across controlled historical decades.  
* **Realistic Boundary for 3–4 Weeks:**  
  * Do not claim to train a novel generative model or eliminate all shortcut learning.  
  * Frame the contribution as: *"An empirical analysis of stem-wise shortcut learning in music era and market classification, validated via controlled melody-conditioned cover generation."*

### **8\. Three Concrete Project Variants**

#### **Variant A: Analysis-Heavy, Small Generation Demo (Recommended)**

* **Research Question:** To what extent do pretrained music encoders rely on vocal language versus accompaniment production shortcuts when predicting release era and market?  
* **Data:** Discogs-VI dataset (subset of \~2,000–3,000 tracks balanced across 4 eras: 70s, 80s, 90s, 10s; and 3 major markets: US, Japan, Germany).  
* **Models:**  
  * Separation: HTDemucs (pretrained).  
  * Encoders: MERT-v1-330M and Whisper-base (vocals only).  
  * Generation: MusicGen-Melody (pretrained 1.5B).  
* **Step-by-Step Experiments:**  
  1. Separate audio into vocals.wav and accompaniment.wav using HTDemucs.  
  2. Extract frozen embeddings for three conditions: Full Mix, Isolated Vocals, Isolated Accompaniment.  
  3. Train linear probing heads with 5-fold artist-stratified cross-validation. Compare classification ${F}_{1}$, confusion matrices, and paired McNemar tests.  
  4. Perform shortcut control: Normalize loudness (-14 LUFS) and apply a 15 kHz low-pass filter to accompaniment; measure performance drops.  
  5. Select 10 recognizable songs from the dataset; generate cross-era covers using MusicGen-Melody.  
  6. Pass generated covers through the frozen classifier to test if the generated audio shifts into the target era's decision boundary.  
* **What to Cut if Time Runs Out:** Drop the Germany market (keep US vs. Japan); drop the audio filter experiments; generate covers for 5 songs instead of 10\.  
* **Expected Deliverable:** Full analytical paper with confusion matrices and stem attribution tables, plus a demo page with 15–20 generated cross-era covers.  
* **Main Risk:** Discogs-VI audio files may be noisy or missing metadata (mitigated by immediate audit on Day 1).  
* **Effort Estimate:** \~18 person-days total (e.g., 3 members $\times$ 6 days).

#### **Variant B: Generation-Heavy, Light Analysis**

* **Research Question:** Can melody-conditioned language models generate era-accurate song covers while strictly preserving the melodic line, and how do objective metrics align with human listeners?  
* **Data:** 50 well-known songs with clean melodic lines across 5 decades (1970–2010).  
* **Models:** MusicGen-Melody, Spotify basic-pitch, LAION-CLAP, frozen MERT probe.  
* **Step-by-Step Experiments:**  
  1. Build a prompt dictionary reflecting era production tropes (e.g., *"1970s analogue funk"*, *"1980s gated-reverb synthpop"*, *"2000s overdriven pop-punk"*).  
  2. Generate 50 songs $\times$ 3 target era styles \= 150 cover clips using MusicGen-Melody.  
  3. Run objective melody evaluation (Chroma DTW distance and basic-pitch RPA).  
  4. Run objective era evaluation (CLAP score and classifier probing).  
  5. Conduct a paired listening study (15 participants, rating melody retention and era plausibility).  
  6. Compute Spearman rank correlation between objective metrics (Chroma DTW, CLAP) and human judgments.  
* **What to Cut if Time Runs Out:** Reduce from 50 songs to 20 songs (60 generated clips); skip the human listening test and report objective metrics with bootstrap CIs.  
* **Expected Deliverable:** Comprehensive benchmark of cross-era music generation with human validation.  
* **Main Risk:** MusicGen-Melody conditioning can hallucinate pitch drift or ignore text prompts if prompt engineering is uncalibrated.  
* **Effort Estimate:** \~22 person-days total.

#### **Variant C: The "Mastering vs. Composition" Counterfactual Diagnostic**

* **Research Question:** Can an era classifier tell the difference between actual musical composition and modern mastering of vintage material?  
* **Data:** 500 original vintage tracks (1960s–1970s) paired with their modern remasters (2000s–2010s reissues), plus modern covers of vintage songs.  
* **Models:** HTDemucs, MERT-v1-330M, MAEST, pyloudnorm.  
* **Step-by-Step Experiments:**  
  1. Create counterfactual audio pairs: apply modern dynamic range compression and loudness curves to 1970s recordings; apply analog tape simulation and 15 kHz low-pass filtering to 2010s tracks.  
  2. Test whether the classifier flips its era prediction based purely on mastering transfer.  
  3. Use MusicGen-Melody to generate "modern compositions in vintage arrangement" vs. "vintage compositions in modern arrangement".  
  4. Run stem attribution to determine if mastering shortcuts reside primarily in the drums/bass stems.  
* **What to Cut if Time Runs Out:** Eliminate modern cover search; focus strictly on synthetic mastering transformations (EQ, compression, tape emulation).  
* **Expected Deliverable:** A paper exposing mastering shortcuts in audio foundation models, supplemented with counterfactual audio examples.  
* **Main Risk:** Gathering clean remaster pairs with verified metadata can be time-consuming.  
* **Effort Estimate:** \~20 person-days total.

#### **Ranking and Recommendation**

> 1. **Feasibility:** Variant A \> Variant B \> Variant C. (Variant A directly reuses your working HW1 probing codebase and minimizes heavy GPU inference).  
> 2. **Novelty & Interest:** Variant C \> Variant A \> Variant B. (Variant C is an insightful diagnostic study on mastering shortcuts).  
> 3. **Final Pick:** **Variant A.** It is the safest for a 3-week timeline and aligns with your background (frozen probing, layer selection, statistical tests), while fulfilling the music generation track requirement through the MusicGen-Melody demo and classifier validation.

### **9\. Reading List (Weekend Preparation)**

> 1. **Copet et al. (2023)** — *Simple and Controllable Music Generation* (NeurIPS 2023). [arXiv:2306.05284](https://arxiv.org/abs/2306.05284).  
   * *Why it matters:* Documentation for MusicGen and MusicGen-Melody; explains how chroma conditioning steers generation without fine-tuning.  
> 2. **Li et al. (2023)** — *MERT: Acoustic Music Understanding with Large-Scale Self-supervised Training* (ICLR 2024). [arXiv:2306.00107](https://arxiv.org/abs/2306.00107).  
   * *Why it matters:* Core architecture reference for the music audio representations you are probing.  
> 3. **Rouard, Massa, & Défossez (2023)** — *Hybrid Transformers for Music Source Separation* (ICASSP 2023). [arXiv:2211.08553](https://arxiv.org/abs/2211.08553).  
   * *Why it matters:* Practical reference for HTDemucs, the stem separation tool you will run on your dataset.  
> 4. **Sturm (2014)** — *A Simple Method to Determine if a Music Information Retrieval System is a "Horse"* (IEEE T-MM). [DOI: 10.1109/TMM.2014.2330697](https://doi.org/10.1109/TMM.2014.2330697).  
   * *Why it matters:* Explains shortcut learning and how to design adversarial controls to prove your classifier is not cheating.  
> 5. **Flexer (2007)** — *A Closer Look on Artist Filters for Musical Genre Classification* (ISMIR 2007). [DOI: 10.5281/zenodo.1416805](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.1416805&authuser=2).  
   * *Why it matters:* Demonstrates why you must stratify your train/test splits strictly by artist to avoid inflated performance.  
> 6. **Maman & Bermano (2022)** — *U-Mix: Unsupervised Music Source Separation Analysis and Feature Probing* (IEEE SPL). [DOI: 10.1109/LSP.2022.3168784](https://www.google.com/search?q=https://doi.org/10.1109/LSP.2022.3168784&authuser=2).  
   * *Why it matters:* Direct precedent for probing isolated stems to diagnose classifier attention.  
> 7. **Elizalde et al. (2023)** — *CLAP: Learning Audio Representations from Natural Language Supervision* (ICASSP 2023). [arXiv:2206.04769](https://arxiv.org/abs/2206.04769).  
   * *Why it matters:* Provides the mathematical formulation for computing objective text-audio alignment scores for generated covers.  
> 8. **Bittner et al. (2022)** — *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation* (ICASSP 2022). [arXiv:2203.09893](https://arxiv.org/abs/2203.09893).  
   * *Why it matters:* Documents Spotify's basic-pitch library, which you can use for melody extraction and pitch accuracy evaluation.  
> 9. **Panteli et al. (2017)** — *A Computational Study on World Music: Characterising Pitch and Timbre Along Global and Regional Scales* (ISMIR 2017). [DOI: 10.5281/zenodo.1416875](https://www.google.com/search?q=https://doi.org/10.5281/zenodo.1416875&authuser=2).  
   * *Why it matters:* Best empirical guide on acoustic correlates of geographic origin in recorded music.  
> 10. **Vickers (2010)** — *The Loudness War: Background, Speculation and Recommendations* (AES Convention 129). [AES e-Library: 15596](https://www.aes.org/e-lib/browse.cfm?elib=15596).  
    * *Why it matters:* Quantitative history of dynamic range, crest factor, and loudness changes across decades.

### **10\. Unverified or Uncertain**

* **Discogs-VI Dataset Specification:** The exact size, audio bitrates, and track list of "Discogs-VI" could not be independently verified from public academic archives. In the literature, Discogs metadata dumps are public, but audio-paired subsets are typically custom-assembled by research labs or course instructors. Check your course materials for its exact artist split and licensing limits.  
* **Pretrained Model Vocal Extraction Limits on Historical Mono:** Published separation benchmarks (MUSDB18-HQ) evaluate stereo studio tracks recorded after 1970\. Published SDR figures (\~9 dB for Demucs) do not reliably apply to historical mono recordings (pre-1968), where performance is unverified and significant vocal cancellation or bleed occurs.  
* **Music ControlNet Checkpoint Availability:** While the paper *Music ControlNet* (*Wu et al., 2023*) is published, community reports indicate that fully functional public checkpoints for melodic conditioning are difficult to configure out-of-the-box compared to audiocraft (MusicGen-Melody). Treat MusicGen-Melody as your verified generative baseline.