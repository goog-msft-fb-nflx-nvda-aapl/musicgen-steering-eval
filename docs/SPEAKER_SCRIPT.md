# Speaker script (TTS verbatim), mid-progress report

## Slide 1

Hello everyone. This is our mid progress report for the music generation track. Our project asks a simple question. When we steer MusicGen toward a musical note by adding a learned direction inside the model, does that steering still hold up under a fair, paired, statistically tested evaluation? We look at three questions. Does steering survive independent evaluators. Which direction method works best. And which layers should we steer.

## Slide 2

Steering means adding a direction vector to the hidden states of a frozen MusicGen while it generates, so the output is pulled toward a concept. Here the concept is the root note of the music. Question one asks whether steering holds up when it is judged by evaluators other than the probe that found the direction. Question two compares four ways of finding the direction, at matched audio fidelity. Question three asks whether the layers where a note is easiest to decode are also the layers where steering works best.

## Slide 3

On the literature side, Music R F M is the closest work. It steers MusicGen with R F M directions, and it is the code base we build on. It does not compare against simpler baselines such as mean difference, logistic regression or L D A. Syn Theory gives labelled synthetic notes, chords, scales and tempo, which we use for training and ground truth. Panda and colleagues compared direction methods for genre and timbre, but not for pitch. Two recent non music papers study layer selection for steering. We are still reading those two in full, so we only claim the gap for the first three rows.

## Slide 4

Here is the setup. We use MusicGen small, with twenty four layers, and MusicGen medium for confirmation. There are twelve root notes and twenty text prompts, with several generation seeds, which gives four hundred and eighty paired clips per cell. The four directions are the top R F M direction, mean difference, logistic regression and shrinkage L D A, all with unit norm. We test two injection arms. One adds the direction at the top eight layers. The other follows the Music R F M recipe, using all layers with probe based weights. Steering strength, called eta, goes from point one five to point six. For evaluation we use a chroma detector, a frozen Mert note probe, and clap for fidelity, with exact paired tests and Holm correction.

## Slide 5

Now the main result for question two. At eta point six, with the Music R F M style arm, the unsteered baseline hits the target note in eight point three percent of clips. Mean difference reaches point nine eight five. R F M reaches point seven eight one. Logistic regression reaches point six seven eight, and L D A only point four four zero. Mean difference beats R F M in all eight arm and eta cells, and logistic and L D A are below R F M in all sixteen, with corrected p values below one in a million. The order is the same on the medium model, which needs larger eta. Interestingly, mean difference has the lowest probe accuracy, yet it steers the most.

## Slide 6

A higher hit rate could simply mean worse audio, so we checked fidelity. We use the clap text audio score and an M M D squared distance between steered and unsteered embeddings. At similar M M D squared, mean difference hits about seventy eight percent, while R F M sits roughly between forty and sixty percent on the same curve. Two controls help. A random direction leaves the hit rate at baseline, between point zero six seven and point one one five. A direction trained on shuffled labels reaches only point one four to point two five, but costs the most clap score. So the gains need note specific content. We have not yet listened to the clips.

## Slide 7

For question one, we need evaluators that do not share the steering probe. The chroma detector is eighty six point three percent accurate on clean labelled notes, and it is biased toward fifths. The frozen Mert note probe reaches ninety point six percent on Syn Theory notes and is independent of MusicGen. The method ranking holds under both evaluators, on small and medium models. Random directions stay at baseline. Where steering fails, the predicted note is usually a fifth, a fourth or a semitone away, and the Mert probe shows the same pattern, so this is not only chroma bias.

## Slide 8

Question three. If the layers with the best probe score were also the best layers to steer, their ranking would correlate with the causal effect of steering. It does not. The mean per note Spearman correlation is about minus point zero one for mean difference and plus point zero seven for R F M, and five of twelve notes are negative. A probe chosen set of eight layers does no better than eight random layers at the lowest eta. A simple rule, using the last eight layers, matches the per note causally chosen set. So in this setup the result is that late layers steer better, not that a per note search adds value. A cheap predictor from activation statistics matches the last eight layers.

## Slide 9

We also ran ablations to test robustness. On the medium model with eta up to two, mean difference reaches point eight three to point nine four, and R F M stays between point four three and point six four. The top R F M direction beats the second and third components in every cell, so it is the fair R F M. Across three training seeds, R F M varies by up to point one one, while mean difference varies little, and the ordering is unchanged. By note, R F M is weak on D sharp and B, while mean difference stays at or above point eight five on every note.

## Slide 10

We want to be clear about what the results do not show. The same eta is not the same perturbation size or audio quality across methods, and clap measures prompt agreement, not note quality. We have not done a listening test, so a drone on the target pitch is not excluded for mean difference. So far we only study one concept family, root notes, on one model family, with two evaluators that both read pitch. Training seed variation is large for R F M, so single seed numbers should not be over read. And we have not tested the large model.

## Slide 11

Here is what comes next. The remaining experiments are mean difference with fewer training clips, a per clip projection strength check for drones, and the weak notes B and D sharp with R F M. Other Syn Theory concepts, such as chords, scales and tempo, will be tried if time allows. After that comes a listening check of the mean difference clips, a full reading of the two layer selection papers, and the figures, tables and final report. All code, logs and result tables are already in the project repository. We are looking for people who are interested in this topic to join and extend it. Thank you, and we are happy to take questions.
