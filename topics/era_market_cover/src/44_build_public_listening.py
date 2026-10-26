"""Build topics/era_market_cover/data_local/listening_public/era_listening/ (static page for GitHub Pages): 2AFC test, neutral vs era prompt, generated accompaniment only (no vocals, no dataset audio).
Clips are renamed to opaque hashes; the trial list (song, era) lives in the page script only. Responses go to a Google Form (config below) plus a CSV fallback download."""
import hashlib, json, os, shutil
R = os.path.dirname(os.path.abspath(__file__)) + "/../../.."; A = R + "/topics/era_market_cover/data_local/listening_public/era_listening/audio"
songs = [s for s in open(R + "/topics/era_market_cover/data_local/listening/selected.txt").read().split("\n") if s]
name = lambda s, e: hashlib.sha1(f"era-listen-{s}-{e}".encode()).hexdigest()[:10] + ".mp3"
for s in songs:
    for e in ("neutral", "1960s", "1980s", "2010s"):
        src = f"{A}/{s}_{e}.mp3"
        if os.path.exists(src): shutil.move(src, f"{A}/{name(s, e)}")
trials = [{"i": f"{k+1:02d}", "era": e, "n": "audio/" + name(s, "neutral"), "p": "audio/" + name(s, e), "song": hashlib.sha1(s.encode()).hexdigest()[:6]} for k, (s, e) in enumerate((s, e) for s in songs for e in ("1960s", "1980s", "2010s"))]
cfg = json.load(open(R + "/topics/era_market_cover/data_local/listening_public/era_listening/form_config.json")) if os.path.exists(R + "/topics/era_market_cover/data_local/listening_public/era_listening/form_config.json") else {"action": "", "entry_id": "", "entry_resp": ""}
html = open(R + "/topics/era_market_cover/src/44_template.html").read().replace("__TRIALS__", json.dumps(trials)).replace("__CFG__", json.dumps(cfg))
open(R + "/topics/era_market_cover/data_local/listening_public/era_listening/index.html", "w").write(html); print(len(trials), "trials")
