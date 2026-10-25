"""Vocal-bleed diagnostics: per-clip vocal share of energy (rms_voc / (rms_voc + rms_acc)) and the Whisper-small language posterior of the ACCOMPANIMENT stem.
If language ID on the accompaniment stem still predicts market, vocal bleed (or non-vocal language cues) remains in the stem.
saves emb/<ds>_vocshare.npy (N,) and emb/<ds>_lid_acc.npy (N,100).  usage: python 08_bleed.py --ds B"""
import argparse, os, numpy as np, torch, librosa, pyloudnorm as pyln
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); a = ap.parse_args()
mid = "openai/whisper-small"; proc = WhisperProcessor.from_pretrained(mid); m = WhisperForConditionalGeneration.from_pretrained(mid).eval().cuda()
tok = proc.tokenizer; ids = tok.convert_tokens_to_ids([f"<|{c}|>" for c in LANGUAGES]); sot = tok.convert_tokens_to_ids("<|startoftranscript|>"); meter = pyln.Meter(SR)
def norm(y):
    l = meter.integrated_loudness(y.astype(np.float64))
    return y if (not np.isfinite(l) or np.abs(y).max() < 1e-4) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
share, lid = [], []
for r in manifest(a.ds):
    z = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz")); v, c = z["vocals"].astype(np.float32), z["accomp"].astype(np.float32)
    rv, rc = np.sqrt((v ** 2).mean()), np.sqrt((c ** 2).mean()); share.append(rv / (rv + rc + 1e-9))
    y16 = librosa.resample(norm(c), orig_sr=SR, target_sr=16000)
    feat = proc(y16, sampling_rate=16000, return_tensors="pt").input_features.cuda()
    with torch.inference_mode(): lg = m(input_features=feat, decoder_input_ids=torch.tensor([[sot]]).cuda()).logits[0, -1]
    lid.append(torch.softmax(lg[ids].float(), -1).cpu().numpy())
np.save(os.path.join(ROOT, "emb", f"{a.ds}_vocshare.npy"), np.array(share, dtype=np.float32)); np.save(os.path.join(ROOT, "emb", f"{a.ds}_lid_acc.npy"), np.array(lid, dtype=np.float32)); print("done", a.ds)
