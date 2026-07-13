import re
import time

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 2000

torch.set_num_threads(1)

start_time = time.perf_counter()

MODEL_NAME = "ibm-granite/granite-speech-4.1-2b-plus"
SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
# Speaker labels and word timestamps are separate model modes; use the official SAA prompt.
SAA_PROMPT = (
    "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding "
    "[Speaker 1]: and [Speaker 2]: tags before speaker turns."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Device: {device}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
model.eval()

audio_path = "media/call_recording1.mp3"
output_path = "media/call_recording1_transcript_plus.txt"


def parse_transcript(text: str) -> list[dict]:
    segments: list[dict] = []
    parts = re.split(r"\[Speaker (\d+)\]:", text)
    for speaker_id, content in zip(parts[1::2], parts[2::2]):
        turn_text = content.strip()
        if turn_text:
            segments.append({"speaker": int(speaker_id), "text": turn_text})
    return segments


@torch.inference_mode()
def transcribe(wav: torch.Tensor) -> str:
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAA_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
    )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, add_special_tokens=False, skip_special_tokens=True).strip()


wav_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
wav = torch.from_numpy(wav_np).unsqueeze(0)
print(f"Loaded {audio_path}: {len(wav_np) / SAMPLE_RATE:.1f}s")

raw = transcribe(wav)
segments = parse_transcript(raw)

if not segments:
    print("Warning: no [Speaker N]: tags found in model output.")
    print(f"Raw output:\n{raw}\n")
    transcript = raw
else:
    lines = [f"Speaker {seg['speaker']}: {seg['text']}" for seg in segments]
    for line in lines:
        print(line)
    transcript = "\n".join(lines)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(transcript)
print(f"\nSaved to {output_path}")
print(f"Total time: {time.perf_counter() - start_time:.1f}s")
