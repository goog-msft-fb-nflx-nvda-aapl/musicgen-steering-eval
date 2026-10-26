"""Second separator for robustness of the stem result: BS-RoFormer (audio-separator package, model_bs_roformer_ep_317_sdr_12.9755.ckpt) vocals / instrumental split of every clip.
writes stems_bsr/<ds>/<id>.npz with the same keys/format as the Demucs stems (vocals, accomp; 24 kHz mono fp16). usage (venv_sep): python 29_separate_bsr.py --ds A [--limit 5]"""
import argparse, os, shutil, time, numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from audio_separator.separator import Separator
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--model", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt"); a = ap.parse_args()
tmp = os.path.join(ROOT, "tmp_sep", a.ds); os.makedirs(tmp, exist_ok=True); out = os.path.join(ROOT, "stems_bsr", a.ds); os.makedirs(out, exist_ok=True)
sep = Separator(output_dir=tmp, output_format="WAV", model_file_dir=os.path.join(ROOT, "sep_models")); sep.load_model(model_filename=a.model)
rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows; t0 = time.time()
def load24(p):
    y, sr = sf.read(p, dtype="float32"); y = y.mean(1) if y.ndim > 1 else y
    return AF.resample(torch.from_numpy(y), sr, SR).numpy() if sr != SR else y
for i, r in enumerate(rows):
    fp = os.path.join(out, r["sample_id"] + ".npz")
    if os.path.exists(fp): continue
    files = sep.separate(r["path"]); paths = [os.path.join(tmp, f) if not os.path.isabs(f) else f for f in files]
    v = [p for p in paths if "(Vocals)" in p][0]; ins = [p for p in paths if "(Instrumental)" in p][0]; voc, acc = load24(v), load24(ins); n = len(sf.read(r["path"], dtype="float32")[0])
    pad = lambda x: np.pad(x, (0, max(0, n - len(x))))[:n]
    np.savez(fp, vocals=pad(voc).astype(np.float16), accomp=pad(acc).astype(np.float16))
    for p in paths: os.remove(p)
    if i % 100 == 0: print(a.ds, i, len(rows), f"{time.time()-t0:.0f}s", flush=True)
print("done", a.ds)
