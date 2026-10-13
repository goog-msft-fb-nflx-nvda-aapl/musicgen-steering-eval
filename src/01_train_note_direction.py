"""RFM note direction on syntheory 'notes', adapted from music-rfm examples/01_train_note_direction.py.
changes vs the example: model size and note are args, outputs go under trained_concepts/<model_tag>/ so runs never collide.
usage: python 01_train_note_direction.py --model small --note "D#"
"""
import argparse, io, json, os, random
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController
from musicrfm.utils import make_json_serializable

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="small", choices=["small", "medium", "large"])
ap.add_argument("--note", default="D#")
ap.add_argument("--n_components", type=int, default=12)
ap.add_argument("--rfm_iters", type=int, default=30)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="./trained_concepts")
a = ap.parse_args()

random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda"
name = f"facebook/musicgen-{a.model}"
tag = f"musicgen_{a.model}"
music_model = MusicgenForConditionalGeneration.from_pretrained(name).to(dev)
music_processor = AutoProcessor.from_pretrained(name)
encodec = EncodecModel.from_pretrained("facebook/encodec_32khz").to(dev)
encodec_proc = AutoProcessor.from_pretrained("facebook/encodec_32khz")
nl = music_model.config.decoder.num_hidden_layers
layers = list(range(-1, -nl, -1))  # same convention as the example (-1..-47 for large's 48 layers)
print("model", name, "decoder layers", nl, "using", len(layers), flush=True)

ctl = MusicGenController(music_model, music_processor, encodec, encodec_proc, control_method="music_rfm",
                         n_components=a.n_components, rfm_iters=a.rfm_iters, batch_size=16)

# decode=False + soundfile: server has no ffmpeg/torchcodec
ds = load_dataset("meganwei/syntheory", "notes")["train"].cast_column("audio", Audio(decode=False))
pos = [x for x in ds if x["root_note_name"] == a.note]
neg = [x for x in ds if x["root_note_name"] != a.note]
n = min(len(pos), len(neg))
pos, neg = random.sample(pos, n), random.sample(neg, n)
i1, i2 = int(0.7 * n), int(0.85 * n)
train, val, test = pos[:i1] + neg[:i1], pos[i1:i2] + neg[i1:i2], pos[i2:] + neg[i2:]
for s in (train, val, test): random.shuffle(s)
print(f"note {a.note}: {n} pos/{n} neg per class; train {len(train)} val {len(val)} test {len(test)}", flush=True)

def dec(x):
    arr, sr = sf.read(io.BytesIO(x["audio"]["bytes"]), dtype="float32", always_2d=True)
    return {"audio": {"array": arr.mean(1), "sampling_rate": sr}}
def feats(s): return torch.cat([ctl.get_audio_features(dec(x)) for x in s], dim=0)
def labs(s): return torch.tensor([1 if x["root_note_name"] == a.note else 0 for x in s]).reshape(-1, 1)
tr_d, va_d, te_d = feats(train), feats(val), feats(test)

pred_accs, dir_accs, results = ctl.compute_directions(
    train_data=tr_d, train_labels=labs(train), val_data=va_d, val_labels=labs(val),
    test_data=te_d, test_labels=labs(test), hidden_layers=layers,
    tuning_metric="auc", pooling="mean", hyperparam_samples=100)
results.update(n_components=a.n_components, rfm_iters=a.rfm_iters, seed=a.seed, model=name, note=a.note,
               n_train=len(train), n_val=len(val), n_test=len(test), layers=layers,
               test_predictor_accs=pred_accs, test_direction_accs=dir_accs)

concept = f"note_{a.note}_ncomp{a.n_components}_seed{a.seed}"
for d in ("directions", "results"): os.makedirs(os.path.join(a.out, tag, d), exist_ok=True)
with open(os.path.join(a.out, tag, "results", f"{concept}.json"), "w") as f:
    json.dump(make_json_serializable(results), f, indent=2)
ctl.save(concept=concept, model_name=tag, path=os.path.join(a.out, tag, "directions"))
print("predictor accs", pred_accs); print("direction accs", dir_accs); print("saved", concept, flush=True)
