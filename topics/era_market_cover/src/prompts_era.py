"""Era prompt texts shared by 07_generate.py (copied verbatim) and 48_lora_musicgen.py."""
ERAS = {"neutral": "a song recording",
        "1960s": "a 1960s pop and soul recording, mono tape sound, live drums, upright bass, brass and organ, warm analog saturation, limited bandwidth",
        "1980s": "a 1980s synth pop recording, gated reverb drums, drum machine, DX7 synthesizer, chorus guitar, glossy digital production",
        "2010s": "a 2010s modern pop production, heavy sub bass, punchy compressed drums, wide polished synths, loud streaming-era mastering"}
ERAS_EXTRA = {"1970s": "a 1970s rock and funk recording, warm analog tape, electric piano, wah guitar, live drums, vinyl-era mix",
              "1990s": "a 1990s pop and hip hop recording, sampled breakbeat drums, early digital synthesizers, CD-era production",
              "2000s": "a 2000s pop and r&b recording, programmed drums, bright digital synths, autotuned pop production, loud compressed mastering",
              "1960s_b": "a 1960s Motown soul record, tambourine, electric bass, brass section, mono mix, tape hiss",
              "1960s_c": "an old vintage recording from the sixties"}

ERAS.update(ERAS_EXTRA)
PROMPTS = {"1960s": ERAS["1960s"], "1970s": ERAS["1970s"], "1980s": ERAS["1980s"], "1990s": ERAS["1990s"], "2000s": ERAS["2000s"], "2010s": ERAS["2010s"]}
