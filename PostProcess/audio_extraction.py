import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from io import BytesIO
from tqdm import tqdm
from pydub import AudioSegment

os.environ["OMP_DISPLAY_ENV"] = "FALSE"
os.environ["KMP_WARNINGS"] = "0"
os.environ["KMP_SETTING"] = "0"

with contextlib.redirect_stderr(open(os.devnull, 'w')):
    import numpy as np
    import parselmouth


def getName(path):
    return os.path.splitext(os.path.basename(path))[0]


def parse_timestamp(ts):
    # ts example: "00:01:23,456"
    parts = re.split(r"[:,]", ts)
    # Expected: [hh, mm, ss, ms]

    if len(parts) != 4:
        raise ValueError(f"Invalid timestamp format: {ts}")

    h, m, s, ms = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt_with_segments(path):
    """Parse SRT → returns (full_text, total_word_count, total_duration, segment_speeds)"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    all_words = []
    segment_speeds = []
    total_duration = 0

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        timestamp_line = next((l for l in lines if "-->" in l), None)
        if not timestamp_line:
            continue

        start_str, end_str = timestamp_line.split(" --> ")

        start_sec = parse_timestamp(start_str)
        end_sec = parse_timestamp(end_str)

        seg_dur = end_sec - start_sec
        if seg_dur <= 0:
            continue

        spoken_lines = [
            line.strip()
            for line in lines
            if not line.strip().isdigit()
               and "-->" not in line
               and line.strip() != ""
        ]

        segment_text = " ".join(spoken_lines)
        words = segment_text.split()
        word_count = len(words)

        if seg_dur > 0 and word_count > 0:
            wps = word_count / seg_dur
            segment_speeds.append(wps)

        all_words.extend(words)
        total_duration = max(total_duration, end_sec)

    full_text = " ".join(all_words)
    total_word_count = len(all_words)

    return full_text, total_word_count, total_duration, segment_speeds


def extract_audio_features(audio_path, srt_path):
    # -------------------------
    # SRT parsing
    # -------------------------
    _, word_count, duration_sec, segment_speeds = parse_srt_with_segments(srt_path)
    speaking_rate_variability = float(np.std(segment_speeds)) if len(segment_speeds) > 1 else 0.0

    # -------------------------
    # File size
    # -------------------------
    file_size_bytes = os.path.getsize(audio_path)

    # -------------------------
    # Load audio using pydub (supports .m4a)
    # -------------------------
    audio_seg = AudioSegment.from_file(audio_path)
    avg_loudness_db = audio_seg.dBFS
    audio_length_sec_audio = audio_seg.duration_seconds
    audio_length_sec = duration_sec if duration_sec > 0 else audio_length_sec_audio

    # Convert to NumPy waveform
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)

    # Normalize to [-1, 1] range
    samples /= (1 << (8 * audio_seg.sample_width - 1))

    # If stereo → convert to mono
    if audio_seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    # -------------------------
    # RMS / loudness variability
    # -------------------------
    sr = audio_seg.frame_rate
    frame = int(sr * 0.03)  # 30ms

    # Frame-wise RMS via rolling window
    if len(samples) > frame:
        kernel = np.ones(frame) / frame
        rms = np.sqrt(np.convolve(samples * samples, kernel, mode='valid'))
        loudness_variability = float(np.std(rms))
    else:
        loudness_variability = 0.0

    # -------------------------
    # Words per minute
    # -------------------------
    wpm = (word_count / audio_length_sec) * 60 if audio_length_sec > 0 else 0

    # -------------------------
    # Pitch via Parselmouth (FAST & accurate)
    # -------------------------
    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    if audio_seg.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    samples /= (1 << (8 * audio_seg.sample_width - 1))

    # Parselmouth requires float64
    samples64 = samples.astype(np.float64)

    # Create Praat Sound object directly from numpy waveform
    snd = parselmouth.Sound(samples64)

    # Extract pitch
    pitch_obj = snd.to_pitch()
    f0_values = pitch_obj.selected_array["frequency"]
    voiced_f0 = f0_values[f0_values > 0]

    avg_pitch_hz = float(np.mean(voiced_f0)) if len(voiced_f0) else 0.0
    pitch_variability = float(np.std(voiced_f0)) if len(voiced_f0) else 0.0

    # -------------------------
    # Normalize variability values for prosody index
    # -------------------------
    norm_pitch_var = min(pitch_variability / 100.0, 1.0)
    norm_speed_var = min(speaking_rate_variability / 1.0, 1.0)
    norm_loud_var = min(loudness_variability / 0.02, 1.0)

    prosody_index = (
            0.6 * norm_pitch_var +
            0.2 * norm_speed_var +
            0.2 * norm_loud_var
    )

    # ------------------------
    # Final Output
    # ------------------------
    return {
            "speed": round(wpm, 2),
            "speaking_rate": round(speaking_rate_variability,2),
            "pitch": round(avg_pitch_hz,2),
            "pitch_variability": round(pitch_variability,2),
            "loudness": round(avg_loudness_db,2),
            "duration": round(audio_length_sec, 2),
            "size": round(file_size_bytes / (1024 * 1024), 2),
            "prosody_index": round(prosody_index, 2),
        }


def extract_all_features(audio_srt_pairs, max_workers=8):
    """
    audio_srt_pairs = [
        ("path/to/audio1.m4a", "path/to/sub1.srt"),
        ("path/to/audio2.m4a", "path/to/sub2.srt"),
        ...
    ]
    Returns: Map of audio_name => feature_dict
    """
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(extract_audio_features, audio_path, srt_path): audio_path
            for audio_path, srt_path in audio_srt_pairs
        }

        pbar = tqdm(total=len(future_map), desc="Processing audio files")

        for future in as_completed(future_map):
            audio_path = future_map[future]
            name = getName(audio_path)

            try:
                feature_data = future.result()
                results[name] = feature_data
            except Exception as e:
                results[name] = {"error": str(e)}

            pbar.update(1)

        pbar.close()

    print(f"Audio extraction completed !")
    return results
