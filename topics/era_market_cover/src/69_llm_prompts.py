"""G06 step 1: LLM-written era prompts. The LLM sees ONLY the decade name and a fixed instruction (no clips, no labels, no results), so the prompts cannot leak dataset information.
8 candidates per decade, sampled (temperature 0.8, top_p 0.95, seeds 0-7). usage: python 69_llm_prompts.py Qwen/Qwen3-8B qwen3_8b -> results/llm_prompts_<tag>.json"""
import sys, json, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import ROOT
mid, tag = sys.argv[1], sys.argv[2]
INSTR = ("Write one text prompt for a text-to-music model. The prompt asks for an instrumental backing track recorded and produced in {dec}. "
         "Describe the instruments, rhythm, and the recording, mixing and mastering style typical of {dec}, in a single sentence of at most 30 words. "
         "Do not mention artists, song titles, lyrics or vocals. Output only the prompt.")
tok = AutoTokenizer.from_pretrained(mid); lm = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval(); out = {}
for dec in ("1960s", "1980s", "2010s"):
    msgs = [{"role": "user", "content": INSTR.format(dec="the " + dec)}]
    kw = {"enable_thinking": False} if "Qwen3" in mid else {}
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kw).to("cuda"); out[dec] = []
    for seed in range(8):
        torch.manual_seed(seed)
        with torch.no_grad(): g = lm.generate(**ids, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=100)
        out[dec].append(tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().strip('"'))
        print(dec, seed, out[dec][-1], flush=True)
json.dump({"model": mid, "instruction": INSTR, "sampling": "temperature 0.8, top_p 0.95, seeds 0-7", "prompts": out}, open(os.path.join(ROOT, "results", f"llm_prompts_{tag}.json"), "w"), indent=1)
