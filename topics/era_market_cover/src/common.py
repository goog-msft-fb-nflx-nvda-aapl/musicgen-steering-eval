import csv, os
ROOT = os.path.expanduser("~/era_market")
DATA = os.path.join(ROOT, "data", "ex")
LABELS = {"A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
          "B": ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]}
SR = 24000

def manifest(ds):
    d = os.path.join(DATA, f"dataset_{ds}")
    rows = list(csv.DictReader(open(os.path.join(d, "manifest.csv"))))
    for r in rows: r["path"] = os.path.join(d, r["audio_path"])
    return rows
