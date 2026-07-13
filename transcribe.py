import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 512

torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model_name = "ibm-granite/granite-4.0-1b-speech"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=dtype).to(device)

audio_path = "media/call_recording.mp3"
output_path = "media/call_recording_transcript.txt"

wav_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
wav = torch.from_numpy(wav_np).unsqueeze(0)
print(f"Loaded {audio_path}: {len(wav_np) / SAMPLE_RATE:.1f}s")

user_prompt = "<|audio|>Please transcribe the following audio to text."
chat = [{"role": "user", "content": user_prompt}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
model_outputs = model.generate(
    **model_inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    num_beams=1,
)

num_input_tokens = model_inputs["input_ids"].shape[-1]
new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
transcript = tokenizer.batch_decode(
    new_tokens, add_special_tokens=False, skip_special_tokens=True
)[0].strip()

print(transcript)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(transcript)
print(f"\nSaved to {output_path}")
