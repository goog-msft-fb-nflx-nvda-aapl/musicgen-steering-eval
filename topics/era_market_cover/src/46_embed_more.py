"""Additional frozen encoders on the same stem conditions as 02b_embed_muq.py (loudness-normalised to -23 LUFS), mean-pooled per layer.
--enc ast  MIT/ast-finetuned-audioset (16 kHz, 3 x 10 s chunks, 13 x 768)   --enc clap  laion/clap-htsat-fused audio tower (48 kHz, 3 x 10 s chunks, projected 512-d embedding)
--enc whisper  openai/whisper-medium encoder (16 kHz, 30 s, 25 x 1024)
saves emb/<ds>_<enc>_<cond>.npy (N, L, D) fp16."""
import argparse, os, numpy as np, torch, torchaudio, soundfile as sf, pyloudnorm as pyln
from common import ROOT, SR, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True); ap.add_argument("--enc", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--conds", nargs="+", default=["mix_norm", "voc_norm", "acc_norm"]); a = ap.parse_args()
meter = pyln.Meter(SR)
def norm(y, t=-23.0):
    if np.abs(y).max() < 1e-4: return y
    l = meter.integrated_loudness(y.astype(np.float64)); return y if not np.isfinite(l) else np.clip(pyln.normalize.loudness(y.astype(np.float64), l, t), -1, 1).astype(np.float32)
def wave(r, c):
    if c.startswith("mix"): y = sf.read(r["path"], dtype="float32")[0]
    else:
        z = np.load(os.path.join(ROOT, "stems", a.ds, r["sample_id"] + ".npz")); y = z["vocals"].astype(np.float32) if c.startswith("voc") else z["accomp"].astype(np.float32)
    return norm(y)
rs = {16000: torchaudio.transforms.Resample(SR, 16000), 48000: torchaudio.transforms.Resample(SR, 48000)}
if a.enc == "ast":
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593"); model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").cuda().eval()
    def emb(y):
        w = rs[16000](torch.from_numpy(y)[None])[0].numpy(); n = len(w) // 3; outs = []
        for k in range(3):
            f = fe(w[k * n:(k + 1) * n], sampling_rate=16000, return_tensors="pt").input_values.cuda()
            o = model(f, output_hidden_states=True); outs.append(torch.stack([h.float().mean(1) for h in o.hidden_states]).squeeze(1))
        return torch.stack(outs).mean(0)
elif a.enc == "whisper":
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium"); enc = WhisperModel.from_pretrained("openai/whisper-medium").encoder.cuda().eval()
    def emb(y):
        f = fe(rs[16000](torch.from_numpy(y)[None])[0].numpy(), sampling_rate=16000, return_tensors="pt").input_features.cuda()
        o = enc(f, output_hidden_states=True); n = min(1500, int(len(y) / SR * 50)); return torch.stack([h.float()[:, :n].mean(1) for h in o.hidden_states]).squeeze(1)
elif a.enc == "clap":
    from transformers import ClapModel, ClapFeatureExtractor
    fe = ClapFeatureExtractor.from_pretrained("laion/clap-htsat-fused"); model = ClapModel.from_pretrained("laion/clap-htsat-fused", use_safetensors=True).cuda().eval()
    def emb(y):
        w = rs[48000](torch.from_numpy(y)[None])[0].numpy(); n = len(w) // 3; outs = []
        for k in range(3):
            f = fe(w[k * n:(k + 1) * n], sampling_rate=48000, return_tensors="pt", truncation="fusion").to("cuda"); g = model.get_audio_features(**f); outs.append((g.pooler_output if hasattr(g, "pooler_output") else g).float())
        return torch.stack(outs).mean(0)
rows = manifest(a.ds); rows = rows[:a.limit] if a.limit else rows
for c in a.conds:
    out = os.path.join(ROOT, "emb", f"{a.ds}_{a.enc}_{c}.npy")
    if os.path.exists(out) and not a.limit: continue
    E = []
    for i, r in enumerate(rows):
        with torch.inference_mode(): e = emb(wave(r, c))
        E.append(e.cpu().numpy().astype(np.float16))
        if i % 300 == 0: print(a.ds, a.enc, c, i, len(rows), E[-1].shape, flush=True)
    E = np.stack(E); assert np.isfinite(E.astype(np.float32)).all(), f"non-finite {c}"
    if not a.limit: np.save(out, E)
    print("saved", c, E.shape, flush=True)
