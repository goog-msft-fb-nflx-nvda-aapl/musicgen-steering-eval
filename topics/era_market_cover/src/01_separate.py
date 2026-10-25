"""HT Demucs separation of every PA1 clip into vocals and accompaniment (drums+bass+other).
input 24 kHz mono wav -> stereo 44.1 kHz -> htdemucs (shifts 1, overlap .25) -> stems resampled to 24 kHz mono, saved fp16.
usage: python 01_separate.py --ds A [--limit N]"""
import argparse, os, time, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from demucs.pretrained import get_model
from demucs.apply import apply_model
from common import ROOT, SR, manifest

ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
model = get_model("htdemucs").cuda().eval(); names = model.sources
out = os.path.join(ROOT, "stems", a.ds); os.makedirs(out, exist_ok=True)
rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows
t0 = time.time()
for i, r in enumerate(rows):
    fp = os.path.join(out, r["sample_id"] + ".npz")
    if os.path.exists(fp): continue
    y, sr = sf.read(r["path"], dtype="float32"); assert sr == SR
    x = AF.resample(torch.from_numpy(y)[None], SR, 44100)
    x = x.repeat(2, 1)[None].cuda()
    ref = x.mean(0)
    with torch.no_grad(): s = apply_model(model, x, shifts=1, split=True, overlap=0.25, progress=False)[0]  # (4, 2, T)
    voc = s[names.index("vocals")]; acc = sum(s[j] for j in range(len(names)) if names[j] != "vocals")
    def down(z): return AF.resample(z.mean(0, keepdim=True), 44100, SR)[0].cpu().numpy().astype(np.float16)
    np.savez(fp, vocals=down(voc), accomp=down(acc))
    if i % 100 == 0: print(i, len(rows), f"{time.time()-t0:.0f}s", flush=True)
print("done", a.ds, f"{time.time()-t0:.0f}s", flush=True)
