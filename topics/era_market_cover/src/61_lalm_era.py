"""Training-free era judge with a large audio-language model (MQAScore / AQAScore-style: use the LALM's own next-token confidence as the score; the course HW3 slides cite MQAScore, not re-read here).
Model: Qwen/Qwen2-Audio-7B-Instruct (safetensors, local HF cache). Two readouts per clip: (a) multiple choice: "Which decade ... ? A) 1960s ... F) 2010s. Answer with the letter only." -> softmax over the six letter logits of the first answer token;
(b) yes/no per decade: "Does this recording sound like it was released in the <decade>? Answer yes or no." -> P(yes) = softmax(yes, no) logits, six decades. Both are reported raw and prior-corrected (subtract the mean log-score of each decade over the reference set, a calibration that uses no labels).
Positive control first: the real accompaniment and mix of the 132 dataset-A VALIDATION clips (accuracy, Spearman of expected decade with true decade). Then generation directories (neutral/1960s/1980s/2010s x seeds, remix NOT used: the accompaniment-only *_acc.wav).
usage: python 61_lalm_era.py --out results/lalm_era.json --gens gen gen_lora ace_gen_c05 ..."""
import argparse, json, os, glob, re, numpy as np, torch, soundfile as sf, librosa
from scipy.stats import spearmanr
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from common import ROOT, SR, LABELS, manifest
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--gens", nargs="*", default=[]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
mid = "Qwen/Qwen2-Audio-7B-Instruct"; proc = AutoProcessor.from_pretrained(mid); model = Qwen2AudioForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval(); tok = proc.tokenizer
dec = LABELS["A"]; letters = "ABCDEF"; lid = [tok.encode(l, add_special_tokens=False)[0] for l in letters]; yes, no = tok.encode("yes", add_special_tokens=False)[0], tok.encode("no", add_special_tokens=False)[0]; Yes, No = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
MC = "Which decade was this music recording most likely released in? A) 1960s B) 1970s C) 1980s D) 1990s E) 2000s F) 2010s. Answer with the letter only."
def first_logits(w16, text):
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}, {"type": "text", "text": text}]}]; prompt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inp = proc(text=prompt, audio=[w16], sampling_rate=16000, return_tensors="pt", padding=True).to("cuda"); inp = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inp.items()}
    with torch.inference_mode(): lg = model(**inp).logits[0, -1].float()
    return lg
def score(w24):
    w16 = librosa.resample(np.asarray(w24, dtype=np.float32), orig_sr=SR, target_sr=16000)[:16000 * 30]
    mc = torch.log_softmax(first_logits(w16, MC)[lid], -1).cpu().numpy(); yn = []
    for d in dec:
        lg = first_logits(w16, f"Does this recording sound like it was released in the {d}? Answer yes or no."); yn.append(float(torch.log_softmax(torch.stack([torch.maximum(lg[yes], lg[Yes]), torch.maximum(lg[no], lg[No])]), 0)[0]))
    return mc, np.array(yn)
rows = manifest("A"); va = [r for r in rows if r["split"] == "validation"]; y = np.array([dec.index(r["label"]) for r in va]); res = {"decades": dec}
def eval_set(name, waves):
    S = [score(w) for w in waves]; mc = np.stack([s[0] for s in S]); yn = np.stack([s[1] for s in S]); out = {}
    for nm, M in (("mc", mc), ("yesno", yn)):
        for cal in (False, True):
            Z = M - M.mean(0, keepdims=True) if cal else M; p = np.exp(Z - Z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); ed = p @ np.arange(6)
            out[f"{nm}{'_cal' if cal else ''}"] = {"acc": float((Z.argmax(1) == y).mean()), "spearman_expdec_vs_true": float(spearmanr(ed, y)[0]), "mean_expdec": float(ed.mean())}
    res[name] = out; print(name, json.dumps({k: {a_: round(b, 3) for a_, b in v.items()} for k, v in out.items()}), flush=True); return mc, yn
sl = lambda r, k: np.load(os.path.join(ROOT, "stems", "A", r["sample_id"] + ".npz"))[k].astype(np.float32)
lim = (lambda L: L[:a.limit]) if a.limit else (lambda L: L)
va = lim(va); y = y[:len(va)]
raw = {}
raw["real_accomp"] = eval_set("real_accomp", [sl(r, "accomp") for r in va]); raw["real_mix"] = eval_set("real_mix", [sf.read(r["path"], dtype="float32")[0] for r in va])
ref_mc = raw["real_accomp"][0].mean(0, keepdims=True); ref_yn = raw["real_accomp"][1].mean(0, keepdims=True); res["gens"] = {}
for d in a.gens:
    ids = [l.split(",")[0] for l in open(os.path.join(ROOT, d, "picked.txt")).read().split("\n")]; out = []
    for s in ids:
        for f in sorted(glob.glob(os.path.join(ROOT, d, f"{s}_*_acc.wav"))):
            m = re.match(r"^(A_[0-9a-f]+)_(.+)_s(\d+)_acc\.wav$", os.path.basename(f))
            if not m or int(m.group(3)) not in a.seeds or m.group(2) not in ("neutral", "1960s", "1980s", "2010s"): continue
            mc, yn = score(sf.read(f, dtype="float32")[0]); out.append({"clip": s, "cond": m.group(2), "seed": int(m.group(3)), "mc": (mc - ref_mc[0]).tolist(), "yn": (yn - ref_yn[0]).tolist(), "mc_raw": mc.tolist(), "yn_raw": yn.tolist()})
        print(d, s, len(out), flush=True)
    res["gens"][d] = out
    json.dump(res, open(a.out, "w"))
json.dump(res, open(a.out, "w"))
