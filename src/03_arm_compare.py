"""paired comparison of steering configs for one note direction on musicgen-small/medium/large.
arms:
  const8   : top-8 layers by val score, weight 1, inject every step (the 02 pilot config)
  musicrfm : all layers, weight w = s^(1/kappa), s = min-max normalised val score, kappa=0.95, per-layer per-step injection prob p=0.3
             (config described in musicrfm iclr 2026 paper, sec. on layer weighting / injection probability; re-implemented here)
every (seed, eta) uses the same torch + python rng seed and same prompts, so arms are paired with the eta=0 baseline.
evaluator: librosa chroma_cqt time-mean; hit = argmax pitch class == target; frac = target share of chroma energy.
usage: python 03_arm_compare.py --model small --note "D#"
"""
import argparse, json, os, random, itertools
import numpy as np, torch, librosa, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small"); ap.add_argument("--note", default="D#")
ap.add_argument("--n_components", type=int, default=12); ap.add_argument("--train_seed", type=int, default=42)
ap.add_argument("--etas", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.6])
ap.add_argument("--gen_seeds", type=int, nargs="+", default=[0, 1, 2, 3])
ap.add_argument("--arms", nargs="+", default=["const8", "musicrfm"])
ap.add_argument("--topk", type=int, default=8); ap.add_argument("--kappa", type=float, default=0.95)
ap.add_argument("--p", type=float, default=0.3); ap.add_argument("--tokens", type=int, default=500)
ap.add_argument("--method", default="rfm", choices=["rfm", "meandiff", "logistic", "lda"])
ap.add_argument("--out", default="./arms")
ap.add_argument("--comp", type=int, default=0, help="rfm eigendirection index used for steering (0 = top; ablation)")
ap.add_argument("--alt_suffix", default="", help="read directions_alt<suffix>/results_alt<suffix> (06 --suffix), e.g. _s43 for a different train seed")
a = ap.parse_args()

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
tgt = NOTES.index(a.note)
PROMPTS = ["a relaxing jazz song with piano", "upbeat electronic dance music with synths", "slow classical piano piece",
           "fast-paced rock song with electric guitar", "acoustic folk song with guitar and soft vocals", "ambient pad music, calm and spacious",
           "funky bass groove with drums", "orchestral cinematic theme with strings", "lo-fi hip hop beat with mellow keys",
           "solo violin melody, expressive", "reggae with offbeat guitar and bass", "dark techno with pulsing kick drum",
           "gentle harp and flute lullaby", "country song with banjo and fiddle", "bright pop song with claps and synth",
           "blues guitar solo over slow shuffle", "latin guitar with percussion", "choir singing a sustained hymn",
           "chiptune video game music", "heavy metal with distorted guitars"]

def add_hooks(dec, directions, layer_w, eta, p, rng):
    """add eta * w_l * unit direction to each layer output; with prob p per layer per forward step.
    own implementation: musicrfm.generation_utils.hook_model breaks on transformers 5 (tensor layer outputs)."""
    hs = []
    for l, w in layer_w.items():
        v = directions[l][a.comp].reshape(1, 1, -1).to(device="cuda", dtype=next(dec.parameters()).dtype) * (eta * w)
        def hook(m, i, o, v=v):
            if p < 1 and rng.random() >= p: return o
            return (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v
        hs.append(dec.model.decoder.layers[l].register_forward_hook(hook))
    return hs

tag = f"musicgen_{a.model}"; name = f"facebook/musicgen-{a.model}"
concept = f"note_{a.note}_ncomp{a.n_components}_seed{a.train_seed}"
tdir = os.path.join("./trained_concepts", tag)
res = json.load(open(os.path.join(tdir, "results", concept + ".json")))
sc = {int(l): m["train_results"]["score"] for l, m in res["layer_metrics"].items()}
lo, hi = min(sc.values()), max(sc.values())
ARMS = {
    "const8": ({l: 1.0 for l in sorted(sorted(sc, key=lambda l: -sc[l])[:a.topk])}, 1.0),
    "musicrfm": ({l: ((sc[l] - lo) / (hi - lo)) ** (1 / a.kappa) for l in sc}, a.p),
}
for k in a.arms: print(k, "layer weights", {l: round(w, 3) for l, w in ARMS[k][0].items()}, "p", ARMS[k][1], flush=True)

model = MusicgenForConditionalGeneration.from_pretrained(os.path.join("models", name.split("/")[-1]) if os.path.isdir(os.path.join("models", name.split("/")[-1])) else name).to("cuda").eval()
proc = AutoProcessor.from_pretrained(name)
enc = EncodecModel.from_pretrained("facebook/encodec_32khz").to("cuda")
encp = AutoProcessor.from_pretrained("facebook/encodec_32khz")
ctl = MusicGenController(model, proc, enc, encp, control_method="music_rfm", n_components=a.n_components, rfm_iters=30, batch_size=16)
ctl.load(concept=concept, model_name=tag, path=os.path.join(tdir, "directions"))
if a.method != "rfm":  # directions + val scores from 06_direction_methods.py
    import pickle
    nn = a.note.replace("#", "sharp")
    ctl.directions = pickle.load(open(f"directions_alt{a.alt_suffix}/{a.method}/{a.model}/note_{nn}.pkl", "rb"))
    sc = {int(l): m["train_results"]["score"] for l, m in json.load(open(f"results_alt{a.alt_suffix}/{a.method}/{a.model}/note_{nn}.json"))["layer_metrics"].items()}
    lo, hi = min(sc.values()), max(sc.values())
    ARMS = {"const8": ({l: 1.0 for l in sorted(sorted(sc, key=lambda l: -sc[l])[:a.topk])}, 1.0),
            "musicrfm": ({l: ((sc[l] - lo) / (hi - lo)) ** (1 / a.kappa) for l in sc}, a.p)}
sr = model.config.audio_encoder.sampling_rate
inp = proc(text=PROMPTS, padding=True, return_tensors="pt").to("cuda")
os.makedirs(a.out, exist_ok=True)

mtag = "" if a.method == "rfm" else a.method + "_"

def run(arm, eta, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    hooks = add_hooks(model.decoder, ctl.directions, ARMS[arm][0], eta, ARMS[arm][1], rng) if arm else []
    with torch.no_grad(): audio = model.generate(**inp, guidance_scale=3.0, max_new_tokens=a.tokens)
    for h in hooks: h.remove()
    out = []
    for i, w in enumerate(audio[:, 0].cpu().numpy()):
        ch = librosa.feature.chroma_cqt(y=w, sr=sr).mean(1)
        out.append(dict(arm=arm or "base", eta=eta if arm else 0.0, seed=seed, prompt=i, argmax=int(ch.argmax()), hit=int(ch.argmax() == tgt),
                        frac=float(ch[tgt] / ch.sum()), chroma=ch.round(4).tolist()))
        if i in (0, 5, 10) and seed == a.gen_seeds[0]:
            sf.write(f"{a.out}/{tag}_{a.note}_{mtag}{arm or 'base'}_eta{eta if arm else 0.0}_s{seed}_p{i}.flac", w, sr, format="FLAC")
    return out

rows = []
for seed in a.gen_seeds:
    rows += run(None, 0.0, seed)
    for arm, eta in itertools.product(a.arms, a.etas):
        rows += run(arm, eta, seed)
    for arm in ["base"] + a.arms:
        r = [x for x in rows if x["seed"] == seed and x["arm"] == arm]
        print(f"seed {seed} {arm}: " + "  ".join(f"eta{e}:{np.mean([x['hit'] for x in r if x['eta']==e]):.2f}" for e in sorted({x['eta'] for x in r})), flush=True)
json.dump(dict(args=vars(a), arms={k: dict(layer_weights=v[0], p=v[1]) for k, v in ARMS.items()}, prompts=PROMPTS, rows=rows),
          open(f"{a.out}/{tag}_{a.note}_arms.json" if a.method == "rfm" else f"{a.out}/{tag}_{a.note}_{a.method}_arms.json", "w"))
print("done", flush=True)
