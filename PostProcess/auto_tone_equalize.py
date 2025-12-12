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
    raw_wav = f"{base}_16.wav"
    sf.write(raw_wav, audio, samplerate=24000, subtype="PCM_24")

    filter_chain = (
        f"atempo={tempo},"
        "silenceremove=start_silence=0.15:start_threshold=-48dB:"
        "stop_silence=0.15:stop_threshold=-48dB,"
        f"{EQ_STR},"
        f"{LIMITER_STR},"
        "aresample=48000:precision=33"
    )

    processed_wav = f"{base}_processed.wav"

    cmd_process = [
        FFMPEG, "-y",
        "-i", raw_wav,
        "-filter:a", filter_chain,
        "-ac", "2",  # convert mono → stereo
        "-c:a", "pcm_s16le",  # 16-bit WAV (lossless)
        processed_wav
    ]
    run(cmd_process)

    # Step 2 — Convert WAV → Apple ALAC (true lossless)
    temp_m4a = f"{base}_tmp.m4a"
    cmd_afconvert = [
        "afconvert",
        "-f", "m4af",
        "-d", "alac",
        processed_wav,
        temp_m4a
    ]
    run(cmd_afconvert)

    final_m4a = f"{base}.m4a"

    cmd_mp4box = [
        "mp4box",
        "-add", temp_m4a,
        "-new", final_m4a
    ]
    run(cmd_mp4box)

    # final = f"{base}.m4a"
    #
    # cmd = [
    #     FFMPEG, "-y",
    #     "-i", raw_wav,
    #     "-filter:a", filter_chain,
    #     "-c:a", "alac",
    #     "-ac", "1",  # mono (recommended for audiobooks)
    #     final
    # ]

    # run(cmd)

    try:
        os.remove(raw_wav)
        os.remove(processed_wav)
        os.remove(temp_m4a)
    except OSError:
        pass


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


