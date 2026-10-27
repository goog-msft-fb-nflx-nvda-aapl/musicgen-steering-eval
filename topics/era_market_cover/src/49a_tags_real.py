"""Discogs-EffNet style tags (400 Discogs genre/style classes) and 1280-d embedding for the ORIGINAL mix of every clip of dataset A and B, used as a genre/style proxy
(the manifest has no genre column). Run in venv_ess (CPU). saves emb/<ds>_dgtags.npy (N,400) and emb/<ds>_dgemb.npy (N,1280) in manifest order, emb/dg_classes.json."""
import json, os, numpy as np, soundfile as sf, librosa
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
from common import ROOT, SR, manifest
M = os.path.join(ROOT, "ess_models"); emb_m = TensorflowPredictEffnetDiscogs(graphFilename=os.path.join(M, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
head = TensorflowPredict2D(graphFilename=os.path.join(M, "genre_discogs400-discogs-effnet-1.pb"), input="serving_default_model_Placeholder", output="PartitionedCall:0")
json.dump(json.load(open(os.path.join(M, "genre_discogs400-discogs-effnet-1.json")))["classes"], open(os.path.join(ROOT, "emb", "dg_classes.json"), "w"))
for ds in ("A", "B"):
    T, E = [], []
    for i, r in enumerate(manifest(ds)):
        y = librosa.resample(sf.read(r["path"], dtype="float32")[0], orig_sr=SR, target_sr=16000); e = emb_m(y); T.append(head(e).mean(0)); E.append(e.mean(0))
        if i % 200 == 0: print(ds, i, flush=True)
    np.save(os.path.join(ROOT, "emb", f"{ds}_dgtags.npy"), np.array(T, dtype=np.float32)); np.save(os.path.join(ROOT, "emb", f"{ds}_dgemb.npy"), np.array(E, dtype=np.float32))
