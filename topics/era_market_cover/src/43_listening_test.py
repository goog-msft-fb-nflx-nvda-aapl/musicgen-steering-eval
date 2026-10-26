"""Builds a local 2AFC listening test (data_local/listening/index.html + audio/, gitignored: contains licensed audio, open it from disk only).
trials: for 8 songs x era prompt X in {1960s, 1980s, 2010s}: two 12 s excerpts of the MusicGen remix (neutral prompt vs prompt X, random A/B order); question 'which excerpt sounds more like the X?'; 24 trials in random order per page load.
participants download a CSV (pid, trial, song, era, chose_prompt_clip, rt). analysis: python 43_listening_test.py --analyze results1.csv [results2.csv ...]  (binomial test vs 0.5 per era, pooled over participants x songs)."""
import argparse, glob, json, os, sys, numpy as np, soundfile as sf
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_local", "listening")
if len(sys.argv) > 2 and sys.argv[1] == "--analyze":
    import csv; from scipy.stats import binomtest; rows = [r for f in sys.argv[2:] for r in csv.DictReader(open(f))]
    for e in ("1960s", "1980s", "2010s"):
        x = [int(r["chose_prompt_clip"]) for r in rows if r["era"] == e]; print(f"{e}: {sum(x)}/{len(x)} chose the {e}-prompt clip ({np.mean(x):.3f}), binomial p={binomtest(sum(x), len(x), .5).pvalue:.4f}" if x else f"{e}: no data")
    sys.exit()
sel = open(os.path.join(D, "selected.txt")).read().split("\n"); os.makedirs(os.path.join(D, "audio"), exist_ok=True); trials = []
for s in sel:
    for e in ("neutral", "1960s", "1980s", "2010s"):
        y, sr = sf.read(os.path.join(D, "raw", f"{s}_{e}_s0_remix.wav"), dtype="float32"); sf.write(os.path.join(D, "audio", f"{s}_{e}.wav"), y[9 * sr:21 * sr], sr)
    for e in ("1960s", "1980s", "2010s"): trials.append({"song": s, "era": e, "neutral": f"audio/{s}_neutral.wav", "prompt": f"audio/{s}_{e}.wav"})
html = """<!doctype html><meta charset=utf-8><title>Era listening test</title><style>body{font:16px system-ui;max-width:640px;margin:2rem auto;padding:0 16px}button{font-size:16px;padding:.6rem 1rem;margin:.3rem}audio{display:block;margin:.4rem 0 1rem}</style>
<h2>Which excerpt sounds more like the given decade?</h2><p>24 pairs of 12 s music excerpts. Listen to A and B (as often as you like), then choose. There are no right or wrong answers you need to know in advance; go by your impression of the era's production and instrumentation.</p>
<p>Participant ID: <input id=pid value=p1> <button onclick=start()>Start</button></p><div id=t></div>
<script>const T=%s;let order=[],i=0,res=[],t0=0;
function shuffle(a){for(let k=a.length-1;k>0;k--){const j=Math.floor(Math.random()*(k+1));[a[k],a[j]]=[a[j],a[k]]}return a}
function start(){order=shuffle(T.map(x=>({...x,flip:Math.random()<.5})));i=0;res=[];show()}
function show(){const d=document.getElementById('t');if(i>=order.length){const c='pid,trial,song,era,chose_prompt_clip,rt_s\\n'+res.join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([c]));a.download='listening_'+document.getElementById('pid').value+'.csv';a.click();d.innerHTML='<p>Done. Send the downloaded CSV file back. Thank you!</p>';return}
const x=order[i],A=x.flip?x.prompt:x.neutral,B=x.flip?x.neutral:x.prompt;t0=Date.now();
d.innerHTML=`<h3>Pair ${i+1} / ${order.length}: which sounds more like the <b>${x.era}</b>?</h3>A<audio controls src="${A}"></audio>B<audio controls src="${B}"></audio><button onclick="ans('A')">A</button><button onclick="ans('B')">B</button>`}
function ans(c){const x=order[i];const chosePrompt=(c==='A')===x.flip?1:0;res.push([document.getElementById('pid').value,i+1,x.song,x.era,chosePrompt,((Date.now()-t0)/1000).toFixed(1)].join(','));i++;show()}
</script>""" % json.dumps(trials)
open(os.path.join(D, "index.html"), "w").write(html); json.dump(trials, open(os.path.join(D, "trials.json"), "w")); print(len(trials), "trials; open", os.path.join(D, "index.html"))
