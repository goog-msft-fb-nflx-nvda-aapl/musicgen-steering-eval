"""Table of results/fd_boot_<set>.json (src/83_fd_bootstrap.py): FD-VGG of generated sets to real PA1 train clips of the same decade; Delta = FD(era prompt) - FD(neutral prompt), 95% song-bootstrap CI; negative = closer to the real decade. Writes results/fd_vgg_table.txt (main decades 1960s/1980s/2010s) and results/fd_vgg_table_extra.txt (extra decades 1970s/1990s/2000s, files fd_boot_trix_*.json)."""
import glob, json, os
for out, decs, pat in (("results/fd_vgg_table.txt", ("1960s", "1980s", "2010s"), "results/fd_boot_[!t]*.json"), ("results/fd_vgg_table_extra.txt", ("1970s", "1990s", "2000s"), "results/fd_boot_trix_*.json")):
    L = [f"set | FD-VGG neutral vs real {' / '.join(decs)} | FD-VGG era prompt vs real same decade | Delta (95% CI); * = CI excludes 0"]
    files = sorted(glob.glob(pat)) + (sorted(glob.glob("results/fd_boot_tri_*.json")) if "[!t]" in pat else [])
    for f in files:
        r = json.load(open(f)); n = os.path.basename(f)[8:-5]
        if not all(f"FD-VGG|{d}" in r for d in decs): continue
        c = {d: r[f"FD-VGG|{d}"] for d in decs}
        L.append(f"{n} | " + " / ".join(f"{c[d]['fd_neutral']:.2f}" for d in c) + " | " + " / ".join(f"{c[d]['delta'] + c[d]['fd_neutral']:.2f}" for d in c) + " | " + " / ".join(f"{c[d]['delta']:+.2f} [{c[d]['ci95'][0]:+.2f},{c[d]['ci95'][1]:+.2f}]" + ("*" if c[d]['ci95'][0] > 0 or c[d]['ci95'][1] < 0 else "") for d in c))
    open(out, "w").write("\n".join(L)); print("\n".join(L))
