"""LoRA fine-tune of MusicGen-Melody (facebook/musicgen-melody) toward era-specific ACCOMPANIMENT.
data: dataset-A TRAIN clips (artist-disjoint from validation). target = EnCodec codes (32 kHz, 4 codebooks) of the Demucs accompaniment stem; conditions = text prompt of the clip's own decade (same prompts as 07_generate.py)
+ chroma of the original mix (same time span). 10 s random crops, delay-pattern cross-entropy per codebook, prompt dropout 10% -> neutral prompt.
adapters on decoder q/k/v/o (self- and cross-attention), r=16. saves lora/<name>/ (peft adapter). usage: python 48_lora_musicgen.py --name v1 --steps 1500"""
import argparse, os, sys, time, numpy as np, torch, soundfile as sf, torchaudio.functional as AF
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
from peft import LoraConfig, get_peft_model
from common import ROOT, SR, manifest, LABELS
sys.argv_backup = sys.argv
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="v1"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--r", type=int, default=16); ap.add_argument("--crop", type=int, default=500); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
import importlib.util; spec = importlib.util.spec_from_file_location("gen", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts_era.py")); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
mid = "facebook/musicgen-melody"; proc = AutoProcessor.from_pretrained(mid)
model = MusicgenMelodyForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda")
rows = [r for r in manifest("A") if r["split"] == "train" and r["label"] in LABELS["A"]]
cdir = os.path.join(ROOT, "lora"); os.makedirs(cdir, exist_ok=True); cf = os.path.join(cdir, "codes_A_train.npy")
if not os.path.exists(cf):
    C = []
    for i, r in enumerate(rows):
        acc = np.load(os.path.join(ROOT, "stems", "A", r["sample_id"] + ".npz"))["accomp"].astype(np.float32); x = AF.resample(torch.from_numpy(acc), SR, 32000)[None, None].to("cuda").bfloat16()
        with torch.no_grad(): c = model.audio_encoder.encode(x).audio_codes
        C.append(c[0, 0].cpu().numpy().astype(np.int16))   # (4, T)
        if i % 200 == 0: print("codes", i, len(rows), C[-1].shape, flush=True)
    T = min(c.shape[1] for c in C); np.save(cf, np.stack([c[:, :T] for c in C]))
codes = np.load(cf).astype(np.int64); print("codes", codes.shape, flush=True)
cfg = model.config.decoder; K = cfg.num_codebooks; PAD = cfg.pad_token_id; BOS = cfg.bos_token_id
lc = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, target_modules=r".*decoder.*(q_proj|k_proj|v_proj|out_proj)", bias="none")
pm = get_peft_model(model, lc); pm.print_trainable_parameters()
for n, p in pm.named_parameters():
    if p.requires_grad: p.data = p.data.float()
# NOTE: the model stays in eval mode while training (no dropout): train mode raised the loss of the pretrained model from 3.9 to 9.6 on the same batch
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 50) * max(0.05, 1 - s / a.steps))
mixcache = {}
def mix32(r):
    if r["sample_id"] not in mixcache:
        mixcache[r["sample_id"]] = AF.resample(torch.from_numpy(sf.read(r["path"], dtype="float32")[0]), SR, 32000).numpy()
        if len(mixcache) > 300: mixcache.pop(next(iter(mixcache)))
    return mixcache[r["sample_id"]]
pm.eval(); t0 = time.time(); ema = None
for step in range(a.steps):
    idx = rng.choice(len(rows), a.bs, replace=False); texts, mixes, tg = [], [], []
    for i in idx:
        r = rows[i]; T = codes.shape[2]; s = int(rng.integers(0, T - a.crop)); tg.append(codes[i, :, s:s + a.crop])
        m = mix32(r); mixes.append(m[int(s / 50 * 32000):int((s + a.crop) / 50 * 32000)]); texts.append("a song recording" if rng.random() < 0.1 else P.PROMPTS[r["label"]])
    L = min(len(m) for m in mixes); inp = proc(text=texts, audio=[m[:L] for m in mixes], sampling_rate=32000, padding=True, return_tensors="pt").to("cuda")
    inp = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
    tgt = torch.from_numpy(np.stack(tg)).cuda()                                   # (B, K, T)
    B, _, T = tgt.shape; d = torch.full((B, K, T + K - 1), PAD, device="cuda", dtype=torch.long)
    for k in range(K): d[:, k, k:k + T] = tgt[:, k]
    dec_in = torch.cat([torch.full((B, K, 1), BOS, device="cuda", dtype=torch.long), d[:, :, :-1]], 2).reshape(B * K, -1)
    out = pm(**inp, decoder_input_ids=dec_in); lg = out.logits.float().reshape(B, K, -1, out.logits.shape[-1])[:, :, -d.shape[2]:]
    lab = d.masked_fill(d == PAD, -100); loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100)
    loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    if step % 25 == 0: print(f"step {step}/{a.steps} loss {loss.item():.3f} ema {ema:.3f} {time.time() - t0:.0f}s", flush=True)
    if (step + 1) % 500 == 0 or step + 1 == a.steps: pm.save_pretrained(os.path.join(cdir, a.name))
print("done", time.time() - t0)
