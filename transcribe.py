import librosa
import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16000
MAX_SPEECH_SECONDS = 60
MAX_NEW_TOKENS = 512
MERGE_GAP_SECONDS = 1.2
MIN_TRANSCRIBE_DURATION = 3.0
END_PAD_SECONDS = 0.5

torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

model_name = "ibm-granite/granite-4.0-1b-speech"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=dtype).to(device)
vad_model = load_silero_vad()


def merge_nearby_segments(
    segments: list[dict[str, float]], max_gap_seconds: float = MERGE_GAP_SECONDS
) -> list[dict[str, float]]:
    if not segments:
        return []

    merged = [dict(segments[0])]
    for segment in segments[1:]:
        gap = segment["start"] - merged[-1]["end"]
        combined_duration = segment["end"] - merged[-1]["start"]
        if gap <= max_gap_seconds and combined_duration <= MAX_SPEECH_SECONDS:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


def detect_speech_segments(wav_np: np.ndarray) -> list[dict[str, float]]:
    wav_tensor = torch.from_numpy(wav_np)
    segments = get_speech_timestamps(
        wav_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        max_speech_duration_s=MAX_SPEECH_SECONDS,
        min_silence_duration_ms=400,
        speech_pad_ms=200,
        return_seconds=True,
    )
    return merge_nearby_segments(segments)


def transcribe_audio(wav_np: np.ndarray) -> str:
    if len(wav_np) == 0:
        return ""

    wav = torch.from_numpy(wav_np).unsqueeze(0)
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
    return tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )[0].strip()


def strip_prefix(transcript: str, prefix: str) -> str:
    transcript = transcript.strip()
    prefix = prefix.strip()
    if not prefix or not transcript:
        return transcript
    if transcript.lower().startswith(prefix.lower()):
        return transcript[len(prefix) :].lstrip(" .,:-")
    return transcript


def slice_audio(wav_np: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    start_sample = int(start_sec * SAMPLE_RATE)
    end_sample = int(min(len(wav_np), end_sec * SAMPLE_RATE))
    return wav_np[start_sample:end_sample]


def transcribe_segment(
    wav_np: np.ndarray,
    segments: list[dict[str, float]],
    index: int,
    prev_canonical_text: str | None,
) -> tuple[str, str]:
    segment = segments[index]
    start_sec = segment["start"]
    end_sec = segment["end"]
    duration = end_sec - start_sec

    canonical_text = transcribe_audio(slice_audio(wav_np, start_sec, end_sec))

    needs_context = index > 0 and (
        duration < MIN_TRANSCRIBE_DURATION or not canonical_text
    )
    if needs_context and prev_canonical_text:
        prev = segments[index - 1]
        context_end = min(len(wav_np) / SAMPLE_RATE, end_sec + END_PAD_SECONDS)
        context_audio = slice_audio(wav_np, prev["start"], context_end)
        context_text = transcribe_audio(context_audio)
        isolated = strip_prefix(context_text, prev_canonical_text)
        if isolated:
            return isolated, canonical_text

    return canonical_text or "", canonical_text


# Load audio
audio_path = "media/call_recording.mp3"
wav_np, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
duration_secs = len(wav_np) / SAMPLE_RATE
print(f"Loaded {audio_path}: {duration_secs:.1f}s")

segments = detect_speech_segments(wav_np)
print(f"Detected {len(segments)} speech segments with VAD\n")

lines = []
prev_canonical_text = None
for i, segment in enumerate(segments, start=1):
    start_sec = segment["start"]
    end_sec = segment["end"]

    print(f"[{i}/{len(segments)}] {start_sec:.1f}s - {end_sec:.1f}s ...")
    text, canonical_text = transcribe_segment(wav_np, segments, i - 1, prev_canonical_text)
    prev_canonical_text = canonical_text
    line = f"[{start_sec:.1f}s - {end_sec:.1f}s] {text}"
    lines.append(line)
    print(f"  {text}\n")

full_transcript = "\n".join(lines)
print("=" * 60)
print("Full transcript:")
print(full_transcript)

output_path = "media/call_recording_transcript.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_transcript)
print(f"\nSaved to {output_path}")
