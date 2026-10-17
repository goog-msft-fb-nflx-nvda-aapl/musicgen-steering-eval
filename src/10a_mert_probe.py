"""second, independent note evaluator: 12-class logistic probe on mean-pooled MERT-v1-95M features of the syntheory 'notes' clips (single-note audio with true labels).
different model family and training signal from the chroma argmax evaluator. layer chosen on a validation split (stratified 70/15/15, seed 42), test acc reported.
saves mert_probe.pkl (layer, scaler mean/std, sklearn model) and mert_probe.json (val/test acc per layer, test confusion). features are cached in mert_feats.npz.
usage: python 10a_mert_probe.py
note: syntheory clips are synthetic renders; the probe is validated on those only, not on generated audio.
"""
import io, json, pickle, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from datasets import Audio, load_dataset
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# the repo only ships pytorch_model.bin and transformers refuses .bin on torch<2.6, so load the official m-a-p checkpoint with torch.load(weights_only=True) (tensors only) into the config-built model
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
cfg = AutoConfig.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
mert = AutoModel.from_config(cfg, trust_remote_code=True)
sd = torch.load(hf_hub_download("m-a-p/MERT-v1-95M", "pytorch_model.bin"), map_location="cpu", weights_only=True)
print(mert.load_state_dict(sd, strict=False), flush=True)
mert = mert.to("cuda").eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
ds = load_dataset("meganwei/syntheory", "notes")["train"].cast_column("audio", Audio(decode=False))
y = np.array([NOTES.index(n) for n in ds["root_note_name"]]); F = []
for i in range(len(ds)):
    arr, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]), dtype="float32", always_2d=True)
    w = AF.resample(torch.tensor(arr.mean(1)), sr, 24000)
    x = fe(w.numpy(), sampling_rate=24000, return_tensors="pt")["input_values"].to("cuda")
    with torch.no_grad(): hs = mert(x, output_hidden_states=True).hidden_states
    F.append(torch.stack([h[0].mean(0) for h in hs]).cpu().numpy())
    if i % 200 == 0: print(i, len(ds), flush=True)
F = np.stack(F).astype(np.float32); np.savez("mert_feats.npz", F=F, y=y)
idx = np.arange(len(y)); tr, rest = train_test_split(idx, test_size=0.3, stratify=y, random_state=42); va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=42)
out = dict(n=[len(tr), len(va), len(te)], val_acc={}, test_acc={})
best = (-1, None)
for l in range(F.shape[1]):
    mu, sd = F[tr, l].mean(0), F[tr, l].std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=500).fit((F[tr, l] - mu) / sd, y[tr]); v = clf.score((F[va, l] - mu) / sd, y[va])
    out["val_acc"][l] = float(v); out["test_acc"][l] = float(clf.score((F[te, l] - mu) / sd, y[te]))
    print("layer", l, "val", round(v, 3), "test", round(out["test_acc"][l], 3), flush=True)
    if v > best[0]: best = (v, (l, mu, sd, clf))
l, mu, sd, clf = best[1]; out["layer"] = l; out["test_acc_best"] = out["test_acc"][l]
out["test_confusion"] = confusion_matrix(y[te], clf.predict((F[te, l] - mu) / sd), labels=range(12)).tolist()
json.dump(out, open("mert_probe.json", "w"), indent=1); pickle.dump(dict(layer=l, mu=mu, sd=sd, clf=clf), open("mert_probe.pkl", "wb"))
print("best layer", l, "val", best[0], "test", out["test_acc_best"], flush=True)
