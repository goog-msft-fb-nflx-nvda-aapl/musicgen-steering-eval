"""Table of results/fd_boot_<set>.json (src/83_fd_bootstrap.py): FD-VGG of generated sets to real PA1 train clips of the same decade; Delta = FD(era prompt) - FD(neutral prompt), 95% song-bootstrap CI; negative = closer to the real decade. Writes results/fd_vgg_table.txt."""
import glob, json, os
L = ["set | FD-VGG neutral vs real 1960s / 1980s / 2010s | FD-VGG era prompt vs real same decade | Delta (95% CI); * = CI excludes 0"]
for f in sorted(glob.glob("results/fd_boot_*.json")):
    r = json.load(open(f)); n = os.path.basename(f)[8:-5]; c = {d: r[f"FD-VGG|{d}"] for d in ("1960s", "1980s", "2010s")}
    L.append(f"{n} | " + " / ".join(f"{c[d]['fd_neutral']:.2f}" for d in c) + " | " + " / ".join(f"{c[d]['delta'] + c[d]['fd_neutral']:.2f}" for d in c) + " | " + " / ".join(f"{c[d]['delta']:+.2f} [{c[d]['ci95'][0]:+.2f},{c[d]['ci95'][1]:+.2f}]" + ("*" if c[d]['ci95'][0] > 0 or c[d]['ci95'][1] < 0 else "") for d in c))
open("results/fd_vgg_table.txt", "w").write("\n".join(L)); print("\n".join(L))
