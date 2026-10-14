"""validate the chroma-argmax note evaluator on labelled audio: syntheory 'notes' clips (single known root note).
checks pitch-class indexing (librosa chroma bin 0 = C) and how often argmax of time-mean chroma_cqt recovers the labelled root.
cpu only. 100 clips per note, fixed seed. reports accuracy per note with wilson 95% CI and the confusion matrix.
this validates the evaluator on synthetic labelled notes only, not on musicgen output or real polyphonic music.
"""
import io, json, random, sys
import numpy as np, librosa, soundfile as sf
from datasets import Audio, load_dataset
from scipy.stats import beta

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
random.seed(0)
ds = load_dataset("meganwei/syntheory", "notes")["train"].cast_column("audio", Audio(decode=False))
names = ds["root_note_name"]
conf = np.zeros((12, 12), int)
for ni, n in enumerate(NOTES):
    idx = [i for i, x in enumerate(names) if x == n]
    for i in random.sample(idx, 100):
        arr, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]), dtype="float32", always_2d=True)
        w = librosa.resample(arr.mean(1), orig_sr=sr, target_sr=32000)  # same sr as musicgen output
        conf[ni, int(librosa.feature.chroma_cqt(y=w, sr=32000).mean(1).argmax())] += 1
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d
out = {"per_note": {}, "overall": None}
for ni, n in enumerate(NOTES):
    k = int(conf[ni, ni]); lo, hi = wilson(k, 100); out["per_note"][n] = dict(acc=k / 100, ci=[lo, hi]); print(n, k / 100, f"[{lo:.3f},{hi:.3f}]")
k = int(np.trace(conf)); lo, hi = wilson(k, 1200); out["overall"] = dict(acc=k / 1200, ci=[lo, hi]); print("overall", k / 1200, f"[{lo:.3f},{hi:.3f}]")
out["confusion"] = conf.tolist()
json.dump(out, open("chroma_validation.json", "w"), indent=1)
