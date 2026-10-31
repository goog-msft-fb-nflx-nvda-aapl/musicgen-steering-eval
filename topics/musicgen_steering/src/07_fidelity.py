"""regenerate the same (note, method, arm, eta, seed, prompt) clips as 03_arm_compare(_v2).py and add per-clip fidelity features:
  - CLAP (laion/clap-htsat-fused via transformers; same htsat-fused family as musicrfm 630k-audioset-fusion-best, not necessarily the identical checkpoint) audio embedding and text-audio cosine score with the generation prompt
  - chroma hit/frac again, compared with the earlier json to check the regeneration reproduces (reported as repro_hit_match)
no audio is kept except a few demo flacs (seed 0, prompts 0/5/10). saves arms_fid/<model>_<note>_<method>_fid.npz.
note: scores are comparable only within this project.
usage: python 07_fidelity.py --note "D#" --method meandiff --gen_seeds 0 1
"""
import argparse, json, os, pickle, random, itertools
import numpy as np, torch, librosa, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel, ClapModel
from musicrfm import MusicGenController

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small"); ap.add_argument("--note", default="D#")
ap.add_argument("--method", default="rfm", choices=["rfm", "meandiff", "logistic", "lda"])
ap.add_argument("--n_components", type=int, default=12); ap.add_argument("--train_seed", type=int, default=42)
ap.add_argument("--etas", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.6])
ap.add_argument("--gen_seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--arms", nargs="+", default=["const8", "musicrfm"])
ap.add_argument("--topk", type=int, default=8); ap.add_argument("--kappa", type=float, default=0.95); ap.add_argument("--p", type=float, default=0.3)
ap.add_argument("--tokens", type=int, default=500); ap.add_argument("--out", default="./arms_fid")
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

def emb(x):  # transformers versions differ in what get_*_features returns
    if isinstance(x, torch.Tensor): return x
    return x.pooler_output if getattr(x, "pooler_output", None) is not None else x[0]

tag = f"musicgen_{a.model}"; name = f"facebook/musicgen-{a.model}"; nn = a.note.replace("#", "sharp")
concept = f"note_{a.note}_ncomp{a.n_components}_seed{a.train_seed}"; tdir = os.path.join("./trained_concepts", tag)
if a.method == "rfm":
    res = json.load(open(os.path.join(tdir, "results", concept + ".json")))
else:
    res = json.load(open(f"results_alt/{a.method}/{a.model}/note_{nn}.json"))
sc = {int(l): m["train_results"]["score"] for l, m in res["layer_metrics"].items()}; lo, hi = min(sc.values()), max(sc.values())
ARMS = {"const8": ({l: 1.0 for l in sorted(sorted(sc, key=lambda l: -sc[l])[:a.topk])}, 1.0),
        "musicrfm": ({l: ((sc[l] - lo) / (hi - lo)) ** (1 / a.kappa) for l in sc}, a.p)}

model = MusicgenForConditionalGeneration.from_pretrained(os.path.join("models", name.split("/")[-1]) if os.path.isdir(os.path.join("models", name.split("/")[-1])) else name).to("cuda").eval(); proc = AutoProcessor.from_pretrained(name)
enc = EncodecModel.from_pretrained("facebook/encodec_32khz").to("cuda"); encp = AutoProcessor.from_pretrained("facebook/encodec_32khz")
ctl = MusicGenController(model, proc, enc, encp, control_method="music_rfm", n_components=a.n_components, rfm_iters=30, batch_size=16)
ctl.load(concept=concept, model_name=tag, path=os.path.join(tdir, "directions"))
if a.method != "rfm": ctl.directions = pickle.load(open(f"directions_alt/{a.method}/{a.model}/note_{nn}.pkl", "rb"))
clap = ClapModel.from_pretrained("laion/clap-htsat-fused").to("cuda").eval(); cproc = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
with torch.no_grad():
    t = emb(clap.get_text_features(**cproc(text=PROMPTS, return_tensors="pt", padding=True).to("cuda"))); t = t / t.norm(dim=-1, keepdim=True)
sr = model.config.audio_encoder.sampling_rate
inp = proc(text=PROMPTS, padding=True, return_tensors="pt").to("cuda"); os.makedirs(a.out, exist_ok=True)

def run(arm, eta, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    hooks = add_hooks(model.decoder, ctl.directions, ARMS[arm][0], eta, ARMS[arm][1], rng) if arm else []
    with torch.no_grad(): audio = model.generate(**inp, guidance_scale=3.0, max_new_tokens=a.tokens)
    for h in hooks: h.remove()
    w = audio[:, 0].float(); rows = []
    with torch.no_grad():
        w48 = AF.resample(w, sr, 48000).cpu().numpy()
        e = emb(clap.get_audio_features(**cproc(audio=list(w48), sampling_rate=48000, return_tensors="pt").to("cuda"))); e = e / e.norm(dim=-1, keepdim=True)
    clapscore = (e * t).sum(-1).cpu().numpy(); E = e.cpu().numpy().astype(np.float16)
    for i, x in enumerate(w.cpu().numpy()):
        ch = librosa.feature.chroma_cqt(y=x, sr=sr).mean(1)
        rows.append(dict(arm=arm or "base", eta=eta if arm else 0.0, seed=seed, prompt=i, hit=int(ch.argmax() == tgt), frac=float(ch[tgt] / ch.sum()), clap=float(clapscore[i])))
        if i in (0, 5, 10) and seed == a.gen_seeds[0] and (arm in (None, "musicrfm")):
            sf.write(f"{a.out}/{tag}_{a.note}_{a.method}_{arm or 'base'}_eta{eta if arm else 0.0}_s{seed}_p{i}.flac", x, sr, format="FLAC")
    return rows, E

rows, embs = [], []
for seed in a.gen_seeds:
    cells = [(None, 0.0)] + [(arm, eta) for arm, eta in itertools.product(a.arms, a.etas)]
    for arm, eta in cells:
        r, E = run(arm, eta, seed); rows += r; embs.append(E)
    print(f"seed {seed} done", flush=True)
# reproducibility vs the earlier chroma-only run
ADIR = "arms" if a.model == "small" else f"arms_{a.model}"
old = json.load(open(f"{ADIR}/{tag}_{a.note}_arms.json" if a.method == "rfm" else f"{ADIR}/{tag}_{a.note}_{a.method}_arms.json"))["rows"]
od = {(o["arm"], o["eta"], o["seed"], o["prompt"]): o["hit"] for o in old}
m = [od[(r["arm"], r["eta"], r["seed"], r["prompt"])] == r["hit"] for r in rows if (r["arm"], r["eta"], r["seed"], r["prompt"]) in od]
print("repro_hit_match", float(np.mean(m)), "n", len(m), flush=True)
np.savez(f"{a.out}/{tag}_{nn}_{a.method}_fid.npz", emb=np.concatenate(embs), rows=json.dumps(rows), repro_hit_match=float(np.mean(m)))
print("saved", flush=True)
