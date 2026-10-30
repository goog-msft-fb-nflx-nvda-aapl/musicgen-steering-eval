"""R04: regenerate the main generation tables from results/*.json only (no GPU, no audio). Calls 56_summarize_gens.py for each (label, eval json, content json) triple that exists and writes results/tables_main.txt.
usage (Mac, from the topic folder): python src/80_make_tables.py"""
import os, subprocess, sys
R = "results"
SETS = {"hand-written CFG3 (seeds 0-2)": ("cfg3_s3", "cfg3_s3"), "delta g3": ("delta_g3", "delta_g3"), "delta g5": ("delta_g5", "delta_g5"), "delta g3 + k250": ("delta_g3k250", "delta_g3k250"),
        "neg-prompt g3": ("neg_g3", "neg_g3"), "neg-prompt g3 + k250": ("neg_g3k250", "neg_g3k250"), "neg-prompt g5": ("neg_g5", "neg_g5"),
        "sampling T1.2 k250": ("sp_t12k250", "sp_t12k250"), "sampling T1.4 k250": ("sp_t14k250", "sp_t14k250"),
        "LoRA no-chroma s1": ("lora_nomels1", "lora_nomels1"), "LoRA no-chroma s2": ("lora_nomels2", "lora_nomels2"),
        "LoRA dropout0.3 s1": ("lora_pd3s1", "lora_pd3s1"), "LoRA dropout0.3 s2": ("lora_pd3s2", "lora_pd3s2"), "LoRA lr3e-4 s2": ("lora_lr3e4s2", "lora_lr3e4s2"), "LoRA v1 + neg-prompt g3": ("lorav1_neg_g3", "lorav1_neg_g3"), "LoRA v1 + delta g3": ("lorav1_delta_g3", "lorav1_delta_g3"), "3-branch base gm3 ge3": ("tri_base_m3e3", "tri_base_m3e3"), "3-branch base gm3 ge5": ("tri_base_m3e5", "tri_base_m3e5"), "3-branch LoRA v1 gm3 ge5": ("tri_lora_m3e5", "tri_lora_m3e5"), "3-branch LoRA v1 gm3 ge7": ("tri_lora_m3e7", "tri_lora_m3e7"), "3-branch LoRA v1 gm3 ge10": ("tri_lora_m3e10", "tri_lora_m3e10"), "3-branch LoRA v1 gm5 ge7": ("tri_lora_m5e7", "tri_lora_m5e7"), "3-branch LoRA v1 gm3 ge7 + k250": ("tri_lora_m3e7k250", "tri_lora_m3e7k250"), "3-branch LoRA v1 gm3 ge15": ("tri_lora_m3e15", "tri_lora_m3e15"), "3-branch LoRA v1 gm3 ge20": ("tri_lora_m3e20", "tri_lora_m3e20"), "3-branch LoRA v1s1 gm3 ge7": ("tri_lora_m3e7v1s1", "tri_lora_m3e7v1s1"), "3-branch LoRA v1s1 gm3 ge10": ("tri_lora_m3e10v1s1", "tri_lora_m3e10v1s1"), "3-branch LoRA v1s2 gm3 ge7": ("tri_lora_m3e7v1s2", "tri_lora_m3e7v1s2"), "3-branch LoRA v1s2 gm3 ge10": ("tri_lora_m3e10v1s2", "tri_lora_m3e10v1s2")}
args = []
for label, (e, c) in SETS.items():
    ef = f"{R}/{e}_eval.json" if not e.startswith("gen_") else f"{R}/{e}.json"
    if not os.path.exists(ef): continue
    cf = f"{R}/{c}_content.json" if c and os.path.exists(f"{R}/{c}_content.json") else None
    args.append(label.replace(" ", "_") + "=" + ef + (":" + cf if cf else ""))
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "56_summarize_gens.py")] + args, capture_output=True, text=True, check=True).stdout
open(f"{R}/tables_main.txt", "w").write(out); print(out)
