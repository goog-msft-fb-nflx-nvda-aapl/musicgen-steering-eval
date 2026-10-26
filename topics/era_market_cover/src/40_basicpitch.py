"""Note-level agreement of generated accompaniment with the ORIGINAL accompaniment stem using basic-pitch (ONNX, venv_bp): note events are transcribed from both, then mir_eval transcription onset-only F1
(50 ms tolerance, pitch tolerance 50 cents) with pitch (note F1) and without (onset F1, pitch ignored by comparing all notes at one pitch). Floor: the same generated notes against another song's original notes.
generators: gen (MusicGen-Melody g=3), mcl_gen (MuseControlLite), gen_textonly; seed 0, 4 prompts. usage: python 40_basicpitch.py --out results/basicpitch.json"""
import argparse, json, os, tempfile, numpy as np, soundfile as sf, mir_eval
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
from common import ROOT, SR
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); tmp = tempfile.mkdtemp(dir=ROOT)
ids = [l.split(",")[0] for l in open(os.path.join(ROOT, "gen", "picked.txt")).read().split("\n")]
def notes(y):
    p = os.path.join(tmp, "x.wav"); sf.write(p, np.asarray(y, dtype=np.float32), SR); ev = predict(p, ICASSP_2022_MODEL_PATH)[2]
    if not ev: return np.zeros((0, 2)), np.zeros(0)
    return np.array([[e[0], e[1]] for e in ev]), 440.0 * 2 ** ((np.array([e[2] for e in ev]) - 69) / 12)
def f1(ref, hyp, pitch):
    (ri, rp), (hi, hp) = ref, hyp
    if len(ri) == 0 or len(hi) == 0: return 0.0
    if not pitch: rp, hp = np.full(len(ri), 440.0), np.full(len(hi), 440.0)
    return float(mir_eval.transcription.precision_recall_f1_overlap(ri, rp, hi, hp, onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None)[2])
O = {s: notes(np.load(os.path.join(ROOT, "stems", "A", s + ".npz"))["accomp"].astype(np.float32)) for s in ids}; print("originals", len(O), flush=True); out = []
for g in ("gen", "mcl_gen", "gen_textonly"):
    for i, s in enumerate(ids):
        other = ids[(i + 1) % len(ids)]
        for e in ("neutral", "1960s", "1980s", "2010s"):
            h = notes(sf.read(os.path.join(ROOT, g, f"{s}_{e}_s0_acc.wav"), dtype="float32")[0])
            out.append({"gen": g, "clip": s, "cond": e, "n_ref": int(len(O[s][0])), "n_hyp": int(len(h[0])), "onset_f1": f1(O[s], h, False), "note_f1": f1(O[s], h, True), "onset_f1_other": f1(O[other], h, False), "note_f1_other": f1(O[other], h, True)})
    print(g, "done", flush=True)
json.dump(out, open(a.out, "w"))
