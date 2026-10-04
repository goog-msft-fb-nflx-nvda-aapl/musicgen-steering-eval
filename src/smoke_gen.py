"""smoke test: musicgen-small baseline generation + throughput, hf transformers port (same one music-rfm hooks)."""
import time, torch, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration

name = "facebook/musicgen-small"
proc = AutoProcessor.from_pretrained(name)
model = MusicgenForConditionalGeneration.from_pretrained(name).to("cuda").eval()
prompts = ["a relaxing jazz song with piano"] * 8
inp = proc(text=prompts, padding=True, return_tensors="pt").to("cuda")
torch.manual_seed(0)
for bs in (1, 8):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        out = model.generate(**{k: v[:bs] for k, v in inp.items()}, do_sample=True, max_new_tokens=500)  # 500 tok = 10 s @ 50 Hz
    torch.cuda.synchronize(); dt = time.time() - t
    print(f"bs={bs} shape={tuple(out.shape)} {dt:.1f}s -> {dt/bs:.2f}s per 10s clip", flush=True)
sf.write("smoke.wav", out[0, 0].cpu().numpy(), model.config.audio_encoder.sampling_rate)
print("peak mem GB", torch.cuda.max_memory_allocated() / 1e9)
