"""pilot: steer musicgen with a trained note direction, score with an evaluator independent of the probe.
paired design: for each (seed, eta) the same prompts + same torch seed; eta=0 is the unsteered baseline (same code path, no hooks).
evaluator: librosa chroma_cqt mean over time, argmax pitch class == target note (note_names C..B).
layers: top-k by val score from the training results json (k=--topk), constant injection, guidance_scale 3.0, 10 s clips.
usage: python 02_pilot_note_steer.py --model small --note "D#"
"""
import argparse, json, os, itertools
import numpy as np, torch, librosa, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController
from musicrfm import generation_utils as gu

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small"); ap.add_argument("--note", default="D#")
ap.add_argument("--topk", type=int, default=8); ap.add_argument("--n_components", type=int, default=12)
ap.add_argument("--train_seed", type=int, default=42)
ap.add_argument("--etas", type=float, nargs="+", default=[0, 0.3, 0.6, 1.0, 1.5])
ap.add_argument("--gen_seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--tokens", type=int, default=500)
ap.add_argument("--out", default="./pilot")
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

tag = f"musicgen_{a.model}"; name = f"facebook/musicgen-{a.model}"
concept = f"note_{a.note}_ncomp{a.n_components}_seed{a.train_seed}"
tdir = os.path.join("./trained_concepts", tag)
res = json.load(open(os.path.join(tdir, "results", concept + ".json")))
sc = {int(l): m["train_results"]["score"] for l, m in res["layer_metrics"].items()}
layers = sorted(sorted(sc, key=lambda l: -sc[l])[:a.topk])
print("steering layers (top-%d by val score):" % a.topk, layers, flush=True)

model = MusicgenForConditionalGeneration.from_pretrained(name).to("cuda").eval()
proc = AutoProcessor.from_pretrained(name)
enc = EncodecModel.from_pretrained("facebook/encodec_32khz").to("cuda")
encp = AutoProcessor.from_pretrained("facebook/encodec_32khz")
ctl = MusicGenController(model, proc, enc, encp, control_method="music_rfm", n_components=a.n_components, rfm_iters=30, batch_size=16)
ctl.load(concept=concept, model_name=tag, path=os.path.join(tdir, "directions"))
sr = model.config.audio_encoder.sampling_rate
norms = {l: float(ctl.directions[l][0].norm()) for l in layers}
print("direction norms", norms, flush=True)

inp = proc(text=PROMPTS, padding=True, return_tensors="pt").to("cuda")
os.makedirs(a.out, exist_ok=True)
rows = []
for seed, eta in itertools.product(a.gen_seeds, a.etas):
    torch.manual_seed(seed)
    hooks = gu.hook_model(model.decoder, ctl.directions, layers, eta) if eta > 0 else {}
    with torch.no_grad():
        audio = model.generate(**inp, guidance_scale=3.0, max_new_tokens=a.tokens)
    gu.clear_hooks(hooks)
    for i, w in enumerate(audio[:, 0].cpu().numpy()):
        ch = librosa.feature.chroma_cqt(y=w, sr=sr).mean(1)
        rows.append(dict(seed=seed, eta=eta, prompt=i, argmax=int(ch.argmax()), hit=int(ch.argmax() == tgt),
                         frac=float(ch[tgt] / ch.sum()), chroma=ch.round(4).tolist()))
        if i in (0, 5, 10): sf.write(f"{a.out}/{tag}_{a.note}_s{seed}_eta{eta}_p{i}.flac", w, sr, format="FLAC")
    r = [x for x in rows if x["seed"] == seed and x["eta"] == eta]
    print(f"seed {seed} eta {eta}: hit {np.mean([x['hit'] for x in r]):.2f} frac {np.mean([x['frac'] for x in r]):.3f}", flush=True)
json.dump(dict(args=vars(a), layers=layers, norms=norms, prompts=PROMPTS, rows=rows), open(f"{a.out}/{tag}_{a.note}_pilot.json", "w"))
