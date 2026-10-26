"""diagnostic: is MERT-v2-30s loading deterministic under transformers 5? loads the model N times, embeds one real clip, reports finiteness and max abs difference of the layer-18 mean embedding vs the first load, and parameters that differ between loads."""
import sys, numpy as np, torch, soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel
from common import manifest
r = manifest("A")[0]; y = sf.read(r["path"], dtype="float32")[0]; proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True); ref = None; refp = None
for k in range(int(sys.argv[1])):
    m = AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True).eval().cuda(); inp = proc(y, sampling_rate=proc.sampling_rate, return_tensors="pt").to("cuda")
    with torch.inference_mode(): h = m(**inp, output_hidden_states=True).hidden_states[18].mean(1).float().cpu().numpy()
    sd = {n: p.detach().float().cpu() for n, p in m.state_dict().items()}; bad = [n for n, p in sd.items() if not torch.isfinite(p).all()]
    if ref is None: ref, refp = h, sd
    diffp = [n for n in sd if n in refp and not torch.equal(sd[n], refp[n])]
    print(k, "finite", bool(np.isfinite(h).all()), "maxdiff vs load0", float(np.abs(h - ref).max()) if np.isfinite(h).all() else None, "nonfinite params", bad[:3], "params differing from load0", diffp[:5], flush=True)
    del m; torch.cuda.empty_cache()
