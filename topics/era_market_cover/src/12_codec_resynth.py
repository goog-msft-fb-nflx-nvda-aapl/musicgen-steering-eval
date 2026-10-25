"""Pass real accompaniment stems through the EnCodec 32 kHz codec used inside MusicGen (4 codebooks = 2.2 kbps at 50 Hz) so real and generated audio share codec artifacts.
24 kHz mono stem -> 32 kHz -> encode/decode (bandwidth 2.2) -> 24 kHz fp16.  saves stems_codec/<ds>/<id>.npz (accomp key).  usage: python 12_codec_resynth.py --ds A"""
import argparse, os, time, numpy as np, torch, torchaudio.functional as AF
from transformers import EncodecModel
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
enc = EncodecModel.from_pretrained("facebook/encodec_32khz").cuda().eval(); print("bandwidths", enc.config.target_bandwidths, flush=True)
bw = 2.2 if 2.2 in enc.config.target_bandwidths else enc.config.target_bandwidths[0]
out = os.path.join(ROOT, "stems_codec", a.ds); os.makedirs(out, exist_ok=True); rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows; t0 = time.time()
for i, r in enumerate(rows):
    fp = os.path.join(out, r["sample_id"] + ".npz")
    if os.path.exists(fp): continue
    y = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz"))["accomp"].astype(np.float32)
    x = AF.resample(torch.from_numpy(y), SR, 32000)[None, None].cuda()
    with torch.no_grad():
        e = enc.encode(x, bandwidth=bw); d = enc.decode(e.audio_codes, e.audio_scales)[0]
    z = AF.resample(d[0, 0].float().cpu(), 32000, SR).numpy()[:len(y)]; z = np.pad(z, (0, len(y) - len(z)))
    np.savez(fp, accomp=z.astype(np.float16))
    if i % 200 == 0: print(i, len(rows), f"{time.time()-t0:.0f}s", f"bw {bw}", flush=True)
print("done", a.ds)
