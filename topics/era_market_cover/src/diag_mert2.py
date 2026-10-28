"""diagnostic 2: nondeterminism within one load, non-persistent buffers, and scale of the effect relative to between-clip differences."""
import numpy as np, torch, soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel
from common import manifest
rows = manifest("A")[:3]; ys = [sf.read(r["path"], dtype="float32")[0] for r in rows]; proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True)
def emb(m, y):
    inp = proc(y, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): return m(**inp, output_hidden_states=True).hidden_states[18].mean(1).float().cpu().numpy()
ms = [AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True).eval().cuda() for _ in range(2)]
m = ms[0]; sdk = set(m.state_dict()); print("buffers not in state_dict:", [n for n, _ in m.named_buffers() if n not in sdk][:10])
a1, a2 = emb(m, ys[0]), emb(m, ys[0]); print("same load, same clip, two passes maxdiff", float(np.abs(a1 - a2).max()))
b = emb(ms[1], ys[0]); print("two loads maxdiff", float(np.abs(a1 - b).max()), "| between-clip maxdiff", float(np.abs(a1 - emb(m, ys[1])).max()), "| norm of embedding", float(np.linalg.norm(a1)))
print("cos two loads", float((a1 * b).sum() / np.linalg.norm(a1) / np.linalg.norm(b)), "cos between clips", float((a1 * emb(m, ys[1])).sum() / np.linalg.norm(a1) / np.linalg.norm(emb(m, ys[1]))))
for n, p in m.named_parameters():
    if "masked_spec_embed" in n or "pos_conv" in n: print(n, tuple(p.shape), float(p.float().abs().mean()))
