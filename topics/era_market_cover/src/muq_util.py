"""MuQ (OpenMuQ/MuQ-large-msd-iter) loader patched for transformers 5: config attn flag and forward hooks that rebuild hidden_states."""
import torch
from muq import MuQ

def load_muq(device="cuda"):
    m = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()
    for mod in m.modules():
        c = getattr(mod, "config", None)
        if c is not None and type(c).__name__ == "EasyDict": c._attn_implementation = "eager"
    conf = m.model.conformer; orig = conf.forward; layers = conf.layers
    def fwd(*a, **k):
        buf = []; hs = [l.register_forward_hook(lambda mod, i, o: buf.append(o[0] if isinstance(o, tuple) else o)) for l in layers]
        try: out = orig(*a, **k)
        finally:
            for h in hs: h.remove()
        last = out["last_hidden_state"] if not isinstance(out, tuple) else out[0]
        return {"hidden_states": tuple(buf) + (last,), "last_hidden_state": last}
    conf.forward = fwd
    return m
