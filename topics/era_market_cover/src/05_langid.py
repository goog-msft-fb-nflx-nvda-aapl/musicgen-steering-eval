"""Whisper language-ID posterior (99 languages) for the vocal stem (norm) and the mix.  saves emb/<ds>_lid_<voc|mix>.npy (N, 99) and lang list.
uses openai/whisper-small via transformers: softmax over language-token logits at the first decoder step.  usage: python 05_langid.py --ds B"""
import argparse, json, os, numpy as np, torch, soundfile as sf, librosa, pyloudnorm as pyln
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
mid = "openai/whisper-small"; proc = WhisperProcessor.from_pretrained(mid); m = WhisperForConditionalGeneration.from_pretrained(mid).eval().cuda()
from transformers.models.whisper.tokenization_whisper import LANGUAGES
tok = proc.tokenizer
lang_toks = [f"<|{c}|>" for c in LANGUAGES]
ids = tok.convert_tokens_to_ids(lang_toks); print(len(ids), lang_toks[:5], flush=True)
sot = tok.convert_tokens_to_ids("<|startoftranscript|>"); meter = pyln.Meter(SR)
def norm(y):
    l = meter.integrated_loudness(y.astype(np.float64))
    return y if (not np.isfinite(l) or np.abs(y).max() < 1e-4) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, -23.0), -1, 1).astype(np.float32)
rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows; res = {"voc": [], "mix": []}
for i, r in enumerate(rows):
    z = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz")); mix = sf.read(r["path"], dtype="float32")[0]
    for k, y in (("voc", norm(z["vocals"].astype(np.float32))), ("mix", mix)):
        y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
        feat = proc(y16, sampling_rate=16000, return_tensors="pt").input_features.cuda()
        with torch.inference_mode(): lg = m(input_features=feat, decoder_input_ids=torch.tensor([[sot]]).cuda()).logits[0, -1]
        res[k].append(torch.softmax(lg[ids].float(), -1).cpu().numpy())
    if i % 200 == 0: print(a.ds, i, len(rows), flush=True)
if not a.limit:
    for k in res: np.save(os.path.join(ROOT, "emb", f"{a.ds}_lid_{k}.npy"), np.array(res[k], dtype=np.float32))
    json.dump([t[2:-2] for t in lang_toks], open(os.path.join(ROOT, "emb", "lid_langs.json"), "w"))
print("done", a.ds)
