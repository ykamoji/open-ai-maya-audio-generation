import numpy as np
import os
import soundfile as sf
import subprocess
import re
import glob


# ================================
# CONFIG
# ================================

DENOISE_STR = "anlmdn=s=6:p=0.0018"
EQ_STR = (
    "highpass=f=80,"
    "equalizer=f=200:t=h:w=350:g=1.4,"     # warmth restore
    "equalizer=f=750:t=h:w=480:g=2.6,"     # chest resonance (fix F1)
    "equalizer=f=3200:t=h:w=2000:g=0.8,"   # clarity band (returns consonant detail)
    "equalizer=f=7200:t=h:w=2600:g=1.6"    # AIR/SPARKLE BOOST (your main sparkle band)

    # "highpass=f=90,"
    # "equalizer=f=220:t=h:w=500:g=0.6,"
    # "equalizer=f=2800:t=h:w=3000:g=0.6,"
    # "equalizer=f=5200:t=h:w=2500:g=0.4"

)

LIMITER_STR = "loudnorm=I=-19.4:TP=-3.0:LRA=11"
#
# LIMITER_STR = "loudnorm=I=-17:TP=-1.5:LRA=6"


FFMPEG = "ffmpeg"
RUBBERBAND = "rubberband-r3"

# ================================
# HELPER TO RUN COMMANDS
# ================================


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,
                   check=True)


# ================================
# MAIN FUNCTION
# ================================
def process_npy(input_path, output_wav, tempo=1.15):
    # load float32 waveform
    print("Loading .npy...")
    audio = np.load(input_path).astype(np.float32)

    base = os.path.splitext(output_wav)[0]

    # save as wav (PCM 16)
    raw_wav = f"{base}_v1.wav"
    sf.write(raw_wav, audio, samplerate=24000, subtype="PCM_24")

    speed = f"{base}_speed.wav"
    # SPEED UP (tempo only, pitch preserved)
    run([FFMPEG, "-y", "-i", raw_wav, "-filter:a", f"atempo={tempo}", speed])

    # Gentle silence reduction (ACX standard)
    trimmed = f"{base}_trimmed.wav"
    run([
        FFMPEG, "-y", "-i", speed,
        "-af",
        "silenceremove=start_silence=0.15:start_threshold=-48dB:stop_silence=0.15:stop_threshold=-48dB",
        trimmed
    ])
    # denoised = f"{base}_denoised.wav"
    # # Light denoise
    # run([FFMPEG, "-y", "-i", trimmed, "-af", "anlmdn=s=7:p=0.003", denoised])

    denoised = trimmed
    eq = f"{base}_eq.wav"
    # ACX EQ (clarity + naturalness)
    run([FFMPEG, "-y", "-i", denoised, "-af", EQ_STR, eq])

    limited = f"{base}_limited.wav"
    # ACX Loudness normalization
    run([
        FFMPEG, "-y", "-i", eq,
        "-af", LIMITER_STR,
        limited
    ])

    final = f"{base}.wav"
    run([FFMPEG, "-y", "-i", limited, "-ar", "48000", final])

    # final = f"{base}.m4a"
    # run([
    #     FFMPEG, "-y",
    #     "-i", limited,
    #     "-ac", "1",  # mono
    #     "-ar", "44100",  # AAC-friendly
    #     "-c:a", "aac",
    #     "-profile:a", "aac_low",
    #     "-b:a", "128k",  # extra stability at 1.25x
    #     "-movflags", "+faststart",
    #     final
    # ])

    os.remove(raw_wav)
    os.remove(speed)
    os.remove(denoised)
    # os.remove(trimmed)
    os.remove(eq)
    os.remove(limited)


# ================================


def getChapter(file):
    return re.search(r"(?i)\bchat?pter\s*#?\s*(\d+)", file).group(1)


def search_files(chapters, inputPath):
    audios = []
    for file in glob.glob(inputPath + "/*/*.npy"):
        number = getChapter(file)
        if number:
            number = int(number)
        if number in chapters and "_meta" not in file:
            audios.append(file)
        # audios.append(file)
    return audios


if __name__ == "__main__":

    chapters = list(range(15, 79))
    audios = search_files(chapters, 'output/audios')
    audios.sort(key=lambda x: int(getChapter(x)))

    audios = audios[1:]
    for audio in audios:
        process_npy(audio, f"output/audios/audiobook_{getChapter(audio)}.wav")

    print(audios)


