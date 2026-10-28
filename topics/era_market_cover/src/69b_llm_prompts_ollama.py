"""G06 step 1 via a local Ollama server (same instruction and sampling settings as 69_llm_prompts.py; weights are the Ollama-packaged quantized models already on the server).
usage: python 69b_llm_prompts_ollama.py <ollama model name> <tag> [port]   -> results/llm_prompts_<tag>.json"""
import sys, json, os, urllib.request
from common import ROOT
mid, tag = sys.argv[1], sys.argv[2]; port = sys.argv[3] if len(sys.argv) > 3 else "11500"
INSTR = ("Write one text prompt for a text-to-music model. The prompt asks for an instrumental backing track recorded and produced in {dec}. "
         "Describe the instruments, rhythm, and the recording, mixing and mastering style typical of {dec}, in a single sentence of at most 30 words. "
         "Do not mention artists, song titles, lyrics or vocals. Output only the prompt.")
out = {}
for dec in ("1960s", "1980s", "2010s"):
    out[dec] = []
    for seed in range(8):
        body = json.dumps({"model": mid, "stream": False, "messages": [{"role": "user", "content": INSTR.format(dec="the " + dec)}], "options": {"temperature": 0.8, "top_p": 0.95, "seed": seed, "num_predict": 100}}).encode()
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/api/chat", body, {"Content-Type": "application/json"}), timeout=300))
        out[dec].append(r["message"]["content"].strip().strip('"')); print(dec, seed, out[dec][-1], flush=True)
json.dump({"model": "ollama:" + mid, "instruction": INSTR, "sampling": "temperature 0.8, top_p 0.95, seeds 0-7", "prompts": out}, open(os.path.join(ROOT, "results", f"llm_prompts_{tag}.json"), "w"), indent=1)
