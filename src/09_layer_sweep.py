"""Q3: decodability-selected layers vs causally steering-effective layers (one note per call, one direction method).
stage 1 (dev prompts 0-9, each twice in one batch of 20): single-layer steering sweep, every layer l in the direction file, weight 1, p=1, eta in --sweep_etas,
  hit = target note is the chroma argmax; dev causal score of a layer = mean hit over etas minus the unsteered dev hit.
stage 2 (held-out prompts 10-19, each twice): multi-layer steering (weight 1, p=1, eta in --val_etas) with three 8-layer sets: probe8 (top-8 by the method's val score, as 07/03 const8),
  causal8 (top-8 by stage-1 dev causal score), rand8 (8 layers drawn with a seeded rng). the layer choice never sees the held-out prompts.
saves layer_sweep/<model>_<note>_<method>_sweep.json with all per-layer numbers (probe val score, dev hits) and the stage-2 cells.
usage: python 09_layer_sweep.py --note C --method meandiff
"""
import argparse, json, os, pickle, random
import numpy as np, torch, librosa
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small"); ap.add_argument("--note", default="C")
ap.add_argument("--method", default="meandiff", choices=["rfm", "meandiff"])
ap.add_argument("--n_components", type=int, default=12); ap.add_argument("--train_seed", type=int, default=42)
ap.add_argument("--sweep_etas", type=float, nargs="+", default=[0.6, 1.5]); ap.add_argument("--val_etas", type=float, nargs="+", default=[0.15, 0.3, 0.45])
ap.add_argument("--tokens", type=int, default=500); ap.add_argument("--out", default="./layer_sweep")
a = ap.parse_args()

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]; tgt = NOTES.index(a.note)
PROMPTS = ["a relaxing jazz song with piano", "upbeat electronic dance music with synths", "slow classical piano piece",
           "fast-paced rock song with electric guitar", "acoustic folk song with guitar and soft vocals", "ambient pad music, calm and spacious",
           "funky bass groove with drums", "orchestral cinematic theme with strings", "lo-fi hip hop beat with mellow keys",
           "solo violin melody, expressive", "reggae with offbeat guitar and bass", "dark techno with pulsing kick drum",
           "gentle harp and flute lullaby", "country song with banjo and fiddle", "bright pop song with claps and synth",
           "blues guitar solo over slow shuffle", "latin guitar with percussion", "choir singing a sustained hymn",
           "chiptune video game music", "heavy metal with distorted guitars"]

def add_hooks(dec, directions, layer_w, eta, p, rng):
    hs = []
    for l, w in layer_w.items():
        v = directions[l][0].reshape(1, 1, -1).to(device="cuda", dtype=next(dec.parameters()).dtype) * (eta * w)
        def hook(m, i, o, v=v):
            if p < 1 and rng.random() >= p: return o
            return (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v
        hs.append(dec.model.decoder.layers[l].register_forward_hook(hook))
    return hs

tag = f"musicgen_{a.model}"; name = f"facebook/musicgen-{a.model}"; nn = a.note.replace("#", "sharp")
concept = f"note_{a.note}_ncomp{a.n_components}_seed{a.train_seed}"; tdir = os.path.join("./trained_concepts", tag)
if a.method == "rfm":
    res = json.load(open(os.path.join(tdir, "results", concept + ".json")))
else:
    res = json.load(open(f"results_alt/{a.method}/{a.model}/note_{nn}.json"))
sc = {int(l): m["train_results"]["score"] for l, m in res["layer_metrics"].items()}; lo, hi = min(sc.values()), max(sc.values())
layers = sorted(sc)

model = MusicgenForConditionalGeneration.from_pretrained(name).to("cuda").eval(); proc = AutoProcessor.from_pretrained(name)
enc = EncodecModel.from_pretrained("facebook/encodec_32khz").to("cuda"); encp = AutoProcessor.from_pretrained("facebook/encodec_32khz")
ctl = MusicGenController(model, proc, enc, encp, control_method="music_rfm", n_components=a.n_components, rfm_iters=30, batch_size=16)
ctl.load(concept=concept, model_name=tag, path=os.path.join(tdir, "directions"))
if a.method != "rfm": ctl.directions = pickle.load(open(f"directions_alt/{a.method}/{a.model}/note_{nn}.pkl", "rb"))
sr = model.config.audio_encoder.sampling_rate
os.makedirs(a.out, exist_ok=True)

def gen_hits(ps, layer_w, eta, seed):
    inp = proc(text=ps * 2, padding=True, return_tensors="pt").to("cuda")
    torch.manual_seed(seed); rng = random.Random(seed)
    hooks = add_hooks(model.decoder, ctl.directions, layer_w, eta, 1.0, rng) if layer_w else []
    with torch.no_grad(): audio = model.generate(**inp, guidance_scale=3.0, max_new_tokens=a.tokens)
    for h in hooks: h.remove()
    return float(np.mean([int(librosa.feature.chroma_cqt(y=x, sr=sr).mean(1).argmax() == tgt) for x in audio[:, 0].float().cpu().numpy()]))

DEV, TEST = PROMPTS[:10], PROMPTS[10:]
base_dev = gen_hits(DEV, None, 0.0, 0); base_test = gen_hits(TEST, None, 0.0, 0)
sweep = {l: {eta: gen_hits(DEV, {l: 1.0}, eta, 0) for eta in a.sweep_etas} for l in layers}
causal = {l: float(np.mean(list(sweep[l].values())) - base_dev) for l in layers}
sets = {"probe8": sorted(sorted(layers, key=lambda l: -sc[l])[:8]), "causal8": sorted(sorted(layers, key=lambda l: -causal[l])[:8]),
        "rand8": sorted(random.Random(100 + tgt).sample(layers, 8))}
cells = {k: {eta: gen_hits(TEST, {l: 1.0 for l in v}, eta, 0) for eta in a.val_etas} for k, v in sets.items()}
out = dict(note=a.note, method=a.method, model=a.model, sweep_etas=a.sweep_etas, val_etas=a.val_etas, base_dev=base_dev, base_test=base_test,
           probe_val_score={str(l): sc[l] for l in layers}, dev_hits={str(l): sweep[l] for l in layers}, causal={str(l): causal[l] for l in layers},
           sets={k: v for k, v in sets.items()}, test_hits={k: {str(e): h for e, h in c.items()} for k, c in cells.items()})
json.dump(out, open(f"{a.out}/{tag}_{nn}_{a.method}_sweep.json", "w"))
print("probe8", sets["probe8"], "causal8", sets["causal8"], "overlap", len(set(sets["probe8"]) & set(sets["causal8"])), flush=True)
print({k: [round(h, 3) for h in c.values()] for k, c in cells.items()}, "base_test", base_test, flush=True)
print("saved", flush=True)
