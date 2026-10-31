"""SECOND EVALUATOR run: same generation as 07_fidelity.py (same clips: seeds, prompts, arms, etas) but each clip is also scored by the MERT-v1-95M note probe from 10a_mert_probe.py
(mean-pooled MERT hidden states at the probe layer -> 12-class logistic). stores per clip: chroma hit (as before), mert_pred, mert_hit, mert prob of the target; clap is not recomputed.
usage: python 10b_second_eval.py --note C --method meandiff --gen_seeds 0 1
the original docstring follows.

regenerate the same (note, method, arm, eta, seed, prompt) clips as 03_arm_compare(_v2).py and add per-clip fidelity features:
  - CLAP (laion/clap-htsat-fused via transformers; same htsat-fused family as musicrfm 630k-audioset-fusion-best, not necessarily the identical checkpoint) audio embedding and text-audio cosine score with the generation prompt
  - chroma hit/frac again, compared with the earlier json to check the regeneration reproduces (reported as repro_hit_match)
no audio is kept except a few demo flacs (seed 0, prompts 0/5/10). saves arms_fid/<model>_<note>_<method>_fid.npz.
note: scores are comparable only within this project.
usage: python 07_fidelity.py --note "D#" --method meandiff --gen_seeds 0 1
"""
import argparse, json, os, pickle, random, itertools
import numpy as np, torch, librosa, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel, AutoModel, AutoConfig, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download
from musicrfm import MusicGenController

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small"); ap.add_argument("--note", default="D#")
ap.add_argument("--method", default="rfm", choices=["rfm", "meandiff", "logistic", "lda"])
ap.add_argument("--n_components", type=int, default=12); ap.add_argument("--train_seed", type=int, default=42)
ap.add_argument("--etas", type=float, nargs="+", default=[0.15, 0.3, 0.45, 0.6])
ap.add_argument("--gen_seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--arms", nargs="+", default=["const8", "musicrfm"])
ap.add_argument("--topk", type=int, default=8); ap.add_argument("--kappa", type=float, default=0.95); ap.add_argument("--p", type=float, default=0.3)
ap.add_argument("--tokens", type=int, default=500); ap.add_argument("--out", default="./arms_eval2")
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
cfgm = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True); mert = AutoModel.from_config(cfgm, trust_remote_code=True)
mert.load_state_dict(torch.load(hf_hub_download("m-a-p/MERT-v1-95M", "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=False); mert = mert.to("cuda").eval()
# transformers 5 HubertEncoder does not return hidden_states for this remote-code model, so capture every encoder layer's output with hooks (12 layers; feature index = layer output index)
_cap = []
def _hk(m, i, o): _cap.append((o[0] if isinstance(o, tuple) else o)[0].mean(0).float().cpu().numpy())
def mert_feats(x):
    _cap.clear()
    with torch.no_grad(): mert(x)
    return np.stack(_cap)
for _l in mert.encoder.layers: _l.register_forward_hook(_hk)
fe = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True); P = pickle.load(open("mert_probe.pkl", "rb"))
sr = model.config.audio_encoder.sampling_rate
inp = proc(text=PROMPTS, padding=True, return_tensors="pt").to("cuda"); os.makedirs(a.out, exist_ok=True)

def run(arm, eta, seed):
    torch.manual_seed(seed); rng = random.Random(seed)
    hooks = add_hooks(model.decoder, ctl.directions, ARMS[arm][0], eta, ARMS[arm][1], rng) if arm else []
    with torch.no_grad(): audio = model.generate(**inp, guidance_scale=3.0, max_new_tokens=a.tokens)
    for h in hooks: h.remove()
    w = audio[:, 0].float(); rows = []
    w24 = AF.resample(w, sr, 24000).cpu().numpy(); feats = []
    for x in w24:
        xi = fe(x, sampling_rate=24000, return_tensors="pt")["input_values"].to("cuda")
        feats.append(mert_feats(xi)[P["layer"]])
    pr = P["clf"].predict_proba((np.stack(feats) - P["mu"]) / P["sd"])
    for i, x in enumerate(w.cpu().numpy()):
        ch = librosa.feature.chroma_cqt(y=x, sr=sr).mean(1)
        rows.append(dict(arm=arm or "base", eta=eta if arm else 0.0, seed=seed, prompt=i, hit=int(ch.argmax() == tgt), ch_pred=int(ch.argmax()), frac=float(ch[tgt] / ch.sum()), mert_pred=int(pr[i].argmax()), mert_hit=int(pr[i].argmax() == tgt), mert_p=float(pr[i, tgt])))
        if i in (0, 5, 10) and seed == a.gen_seeds[0] and (arm in (None, "musicrfm")):
            sf.write(f"{a.out}/{tag}_{a.note}_{a.method}_{arm or 'base'}_eta{eta if arm else 0.0}_s{seed}_p{i}.flac", x, sr, format="FLAC")
    return rows, None

rows = []
for seed in a.gen_seeds:
    cells = [(None, 0.0)] + [(arm, eta) for arm, eta in itertools.product(a.arms, a.etas)]
    for arm, eta in cells:
        r, _ = run(arm, eta, seed); rows += r
    print(f"seed {seed} done", flush=True)
np.savez(f"{a.out}/{tag}_{nn}_{a.method}_eval2.npz", rows=json.dumps(rows))
print("saved", flush=True)
