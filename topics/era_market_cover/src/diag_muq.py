"""diagnostic: MuQ determinism across loads (same check as diag_mert_fixed.py), to rule out the same uninitialized-buffer problem."""
import numpy as np, torch, soundfile as sf
from common import manifest
from muq_util import load_muq
y = sf.read(manifest("A")[0]["path"], dtype="float32")[0]; outs = []
for k in range(3):
    m = load_muq(); sdk = set(m.state_dict()); nb = [n for n, _ in m.named_buffers() if n not in sdk]
    with torch.inference_mode(): o = m(torch.from_numpy(y)[None].cuda(), output_hidden_states=True)["hidden_states"][9].float().mean(1).cpu().numpy()
    outs.append(o); print(k, "finite", bool(np.isfinite(o).all()), "maxdiff vs load0", float(np.abs(o - outs[0]).max()), "non-persistent buffers:", nb[:6], flush=True); del m; torch.cuda.empty_cache()
