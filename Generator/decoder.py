import torch
import os
import soundfile as sf
import numpy as np
import glob
import subprocess
import re
from pathlib import Path
from Emotions.utils import getDevice, clear_cache

CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
WIN = 28
HOP = 7


def extract_snac_codes(token_ids: list) -> list:
    """Extract SNAC codes from generated tokens."""
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)

    snac_codes = [
        token_id for token_id in token_ids[:eos_idx]
        if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]

    return snac_codes


def unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):
        slots = snac_tokens[i * 7:(i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])

    return [l1, l2, l3]


def saveAudio(outputPath, audio_frames, title):
    file = os.path.join(outputPath, f"{title}.npy")
    Path(outputPath).mkdir(parents=True, exist_ok=True)
    np.save(file, audio_frames)
    print(f"Saved {title}.")
    # sf.write(outputPath + f'{title}.wav', audio_frames, samplerate=24000)


def decode_audio(generated_outputs, snac_model):

    # Extract SNAC audio tokens
    generated_snac_tokens = []
    for tokens in generated_outputs:
        generated_snac_tokens.extend(extract_snac_codes(tokens))

    N = len(generated_snac_tokens)
    audio_frames = []
    for i in range(0, N - WIN + 1,  HOP):

        window_tokens = generated_snac_tokens[i: i + WIN]

        levels = unpack_snac_from_7(window_tokens)

        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=getDevice()).unsqueeze(0)
            for level in levels
        ]

        with torch.inference_mode():
            z_q = snac_model.quantizer.from_codes(codes_tensor)
            audio = snac_model.decoder(z_q)[0, 0].float().cpu().numpy()

        codes_tensor = None
        z_q = None
        clear_cache()

        L = len(audio) // 4
        center_frame = audio[L: 2 * L]
        audio_frames.append(center_frame)

    return np.concatenate(audio_frames).astype(np.float32)


def create_audio(generated_outputs, snac_model, audio_path, title):

    audio_frames = decode_audio(generated_outputs, snac_model)

    saveAudio(audio_path, audio_frames, title)


def writing_audio(chapters, outputPath):
    audios = []
    for file in glob.glob(outputPath + "/*/*.npy"):
        number = re.search(r"(?i)\bchat?pter\s*#?\s*(\d+)", file).group(1)
        if "partial" not in file and number in chapters:
            audios.append(file)
        # audios.append(file)

    audios.sort(key=os.path.getmtime)
    audiobook = []
    for audio in audios[-1:]:
        audiobook.append(np.load(audio))

    if len(audiobook) == 1:
        audiobook = audiobook[0]
    else:
        audiobook = np.concatenate(audiobook)
    final_audio_path = outputPath + f'audiobook.wav'
    sf.write(final_audio_path, audiobook, 24000,  subtype="FLOAT")
    return final_audio_path


# ======================
# CONFIGURATION
# ======================

TEMPO = 1.3                # speed-up (keeps slow style)
DENOISE_STR = "anlmdn=s=7:p=0.003"
EQ_STR = (
    "equalizer=f=3000:t=h:w=1500:g=2," 
    "equalizer=f=5000:t=h:w=1200:g=1,"
    "equalizer=f=180:t=h:w=120:g=-1"
)
LIMITER_STR = "loudnorm=I=-19:TP=-3:LRA=11"
TARGET_SR = 48000            # Final polished output

RUBBERBAND = "rubberband-r3"   # Ensure installed
FFMPEG = "ffmpeg"           # Ensure installed



def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ffmpeg_filter(input_wav, output_wav, filters):
    cmd = [
        FFMPEG, "-y",
        "-i", input_wav,
        "-af", filters,
        output_wav
    ]
    run(cmd)


def ffmpeg_resample(input_wav, output_wav, sr):
    cmd = [
        FFMPEG, "-y",
        "-i", input_wav,
        "-ar", str(sr),
        # "-af", "aresample=resampler=soxr",
        output_wav
    ]
    run(cmd)


def upgrade(input_wav, output_wav):

    base = os.path.splitext(input_wav)[0]

    # Resample to 48kHz (soxr HQ)
    voice48 = f"{base}_pre48.wav"
    run([FFMPEG, "-y", "-i", input_wav, "-ar", "48000", voice48])

    # Speedup (Rubberband)
    fast = f"{base}_fast.wav"
    run([RUBBERBAND, "-T", f"{TEMPO}", "-p", "0", "-3", voice48, fast])

    # Gentle silence reduction (ACX standard)
    trimmed = f"{base}_trimmed.wav"
    run([
        FFMPEG, "-y", "-i", fast,
        "-af",
        "silenceremove=start_silence=0.30:start_threshold=-45dB:"
        "stop_silence=0.30:stop_threshold=-45dB",
        trimmed
    ])

    # Light denoise
    denoised = f"{base}_denoised.wav"
    run([FFMPEG, "-y", "-i", trimmed, "-af", "anlmdn=s=7:p=0.003", denoised])

    # ACX EQ (clarity + naturalness)
    eq = f"{base}_eq.wav"
    run([
        FFMPEG, "-y", "-i", denoised,
        "-af",
        "equalizer=f=3000:t=h:w=1500:g=2,"
        "equalizer=f=5000:t=h:w=1200:g=1,"
        "equalizer=f=180:t=h:w=120:g=-1",
        eq
    ])

    # ACX Loudness normalization
    limited = f"{base}_limited.wav"
    run([
        FFMPEG, "-y", "-i", eq,
        "-af", "loudnorm=I=-19:TP=-3:LRA=11",
        limited
    ])

    # Final 48kHz master
    run([FFMPEG, "-y", "-i", limited, "-ar", "48000", output_wav])

    for f in [trimmed, voice48, fast, denoised, eq, limited]:
        try:
            os.remove(f)
        except Exception as e:
            pass


if __name__ == '__main__':

    outputPath = 'output/audios/'
    chapters = list(range(15, 79))

    final_audio_path = writing_audio(chapters, outputPath)

    audioPath = outputPath + 'audiobook.wav'

    upgrade(audioPath, outputPath+'audiobook_v2.wav')
