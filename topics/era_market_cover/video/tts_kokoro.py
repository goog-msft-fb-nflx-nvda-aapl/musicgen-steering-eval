"""Narration with Kokoro-82M voice af_heart (same voice as the topic-1 video). usage: tts_kokoro.py <outdir>"""
import sys, glob, os, numpy as np, soundfile as sf
from kokoro import KPipeline
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
p = KPipeline(lang_code='a')
for f in sorted(glob.glob('script/*.txt')):
    txt = open(f).read().strip()
    seg = [a for _, _, a in p(txt, voice='af_heart', speed=1.0, split_pattern=r'(?<=[.?!])\s+')]
    gap = np.zeros(int(24000 * 0.25), dtype=np.float32)
    audio = np.concatenate([x for a in seg for x in (np.asarray(a, dtype=np.float32), gap)])
    sf.write(f'{out}/{os.path.basename(f)[:-4]}.wav', audio, 24000); print(f, len(audio) / 24000, flush=True)
