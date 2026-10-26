"""MERT-v2-30s loader that fixes a transformers-5 loading bug: the non-persistent buffer embed_positions.inv_freq (rotary frequencies) is left UNINITIALIZED (random memory, sometimes denormal/NaN),
so every process got slightly different embeddings (cos 0.9855 between two loads of the same clip vs 0.868 between different clips) and occasionally all-NaN output. We recompute it exactly as RotaryEmbedding.__init__ does."""
import torch
from transformers import AutoFeatureExtractor, AutoModel
def load_mert(cuda=True):
    proc = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True); model = AutoModel.from_pretrained("m-a-p/MERT-v2-30s", trust_remote_code=True).eval()
    ep = model.embed_positions; hd = model.config.hidden_size // model.config.num_attention_heads
    inv = 1.0 / (model.config.rotary_embedding_base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)); ep.inv_freq = inv.to(ep.inv_freq.device); ep._cos = None; ep._sin = None; ep._cache_device = None; ep._sequence_length = 0
    assert torch.isfinite(ep.inv_freq).all() and abs(float(ep.inv_freq[0]) - 1.0) < 1e-6
    return proc, (model.cuda() if cuda else model)
