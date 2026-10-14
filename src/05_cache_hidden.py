"""cache mean-pooled decoder hidden states (layers -1..-(L-1), same convention as 01) for every syntheory 'notes' clip.
saves hidden_<model>.npz: H [N, L-1, d] float16, layer ids, and metadata. uses musicrfm's own extraction function so features match 01.
usage: python 05_cache_hidden.py --model small
"""
import argparse, io, numpy as np, torch, soundfile as sf
from datasets import Audio, load_dataset
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel
from musicrfm import MusicGenController
from musicrfm import direction_utils as du

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="small"); ap.add_argument("--chunk", type=int, default=128)
a = ap.parse_args()
name = f"facebook/musicgen-{a.model}"
model = MusicgenForConditionalGeneration.from_pretrained(name).to("cuda").eval()
proc = AutoProcessor.from_pretrained(name)
enc = EncodecModel.from_pretrained("facebook/encodec_32khz").to("cuda"); encp = AutoProcessor.from_pretrained("facebook/encodec_32khz")
ctl = MusicGenController(model, proc, enc, encp, control_method="music_rfm", n_components=12, rfm_iters=30, batch_size=16)
nl = model.config.decoder.num_hidden_layers; layers = list(range(-1, -nl, -1))
ds = load_dataset("meganwei/syntheory", "notes")["train"].cast_column("audio", Audio(decode=False))
N = len(ds); out = None
def dec(i):
    arr, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]), dtype="float32", always_2d=True)
    return {"audio": {"array": arr.mean(1), "sampling_rate": sr}}
for s in range(0, N, a.chunk):
    codes = torch.cat([ctl.get_audio_features(dec(i)) for i in range(s, min(N, s + a.chunk))], dim=0)
    hs = du.get_hidden_states_music(codes, model.decoder, proc, layers, 16, pooling="mean")
    blk = torch.stack([hs[l] for l in layers], 1).numpy().astype(np.float16)  # [n, L-1, d]
    if out is None: out = np.zeros((N, blk.shape[1], blk.shape[2]), np.float16)
    out[s:s + len(blk)] = blk
    print(s, N, flush=True)
np.savez(f"hidden_{a.model}.npz", H=out, layers=np.array(layers), root_note_name=np.array(ds["root_note_name"]),
         octave=np.array(ds["octave"]), midi_program_name=np.array(ds["midi_program_name"]))
print("saved", out.shape)
