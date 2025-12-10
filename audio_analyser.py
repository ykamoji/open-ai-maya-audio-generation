import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pyloudnorm import Meter
import numpy as np
import pyloudnorm as pyln
import librosa
from pydub import AudioSegment

# ----------------------------------------------------
# CONFIG FOR SILENCE ANALYSIS
# ----------------------------------------------------
SILENCE_CHUNK_MS = 80
MA_WINDOW = 10
RELATIVE_DIP_DB = 6
MIN_SILENCE_MS = 140


# ----------------------------------------------------
def find_relative_dips(wav_path,
                       chunk_ms=80,
                       ma_window=10,
                       relative_dip_db=6,
                       min_silence_ms=140):
    y, sr = librosa.load(wav_path, sr=None)
    chunk = int(sr * (chunk_ms/1000.0))
    frame_db = []
    for i in range(0, len(y), chunk):
        frame = y[i:i+chunk]
        rms = np.sqrt(np.mean(frame**2)+1e-12)
        db = librosa.amplitude_to_db(np.array([rms]), ref=np.max)[0]
        frame_db.append(db)
    frame_db = np.array(frame_db)

    # diagnostics
    q = np.percentile(frame_db, [0,10,25,50,75,90,100])
    print("FRAME DB STATS (min,10,25,50,75,90,max):", np.round(q,2))
    med = np.median(frame_db)
    for d in (4,6,8,10,12):
        print(f"frames below median-{d}dB:", np.sum(frame_db < (med - d)))

    # moving average
    def moving_avg(a, n):
        if len(a) < n:
            return np.full_like(a, np.mean(a))
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        out = np.empty_like(a)
        out[:n-1] = np.mean(a[:n-1])
        out[n-1:] = ret[n-1:] / n
        return out
    ma = moving_avg(frame_db, ma_window)

    thresh = ma - relative_dip_db
    silent_mask = frame_db < thresh

    # group frames
    segments = []
    s = None
    for i, m in enumerate(silent_mask):
        if m and s is None:
            s = i
        if (not m) and s is not None:
            segments.append((s, i))
            s = None
    if s is not None:
        segments.append((s, len(silent_mask)))

    # convert and filter by duration
    out_segments = []
    for a,b in segments:
        dur_ms = (b-a)*chunk_ms
        if dur_ms >= min_silence_ms:
            out_segments.append(((a*chunk_ms)/1000.0, (b*chunk_ms)/1000.0, dur_ms))
    print("Detected silent segments (s,e,duration_ms):")
    for seg in out_segments:
        print(" ", seg)
    return out_segments, frame_db, ma


def compress_dips_to_silence(wav_path, out_path,
                             chunk_ms=80,
                             ma_window=10,
                             relative_dip_db=6,
                             min_silence_ms=140,
                             keep_ms=120):
    # find segments
    segs, frame_db, ma = find_relative_dips(wav_path, chunk_ms, ma_window, relative_dip_db, min_silence_ms)
    if len(segs) == 0:
        print("No segments to compress. Try lowering thresholds.")
        # still copy original
        data, sr = sf.read(wav_path)
        sf.write(out_path, data, sr)
        return

    # load audio with pydub and rebuild
    audio = AudioSegment.from_wav(wav_path)
    out = AudioSegment.silent(duration=0)
    last = 0
    for (s, e, dur) in segs:
        s_ms = int(round(s*1000))
        e_ms = int(round(e*1000))
        out += audio[last:s_ms]
        out += AudioSegment.silent(duration=keep_ms)
        last = e_ms
    out += audio[last:]
    out.export(out_path, format="wav")
    print("Wrote compressed file:", out_path)


def load_audio(path):
    audio, sr = librosa.load(path, sr=None, mono=True)
    return audio, sr

def high_freq_energy(y, sr, low=10000, high=16000):
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean(fft[mask]**2)))

def spectral_flatness(y):
    S = np.abs(librosa.stft(y))
    flatness = librosa.feature.spectral_flatness(S=S)
    return float(np.mean(flatness))

def spectral_flatness_high(y, sr, low=8000):
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mask = freqs >= low
    if mask.sum() == 0:
        return 0.0
    S_high = S[mask]
    flatness = librosa.feature.spectral_flatness(S=S_high)
    return float(np.mean(flatness))

def spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid))

def spectral_rolloff(y, sr, roll_percent=0.95):
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=roll_percent)
    return float(np.mean(roll))

def onset_strength(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.mean(onset_env))

def group_delay_variance(y, sr):
    S = librosa.stft(y)
    _, phase = librosa.magphase(S)
    phase_unwrap = np.unwrap(np.angle(phase))
    gd = -np.diff(phase_unwrap, axis=0)
    return float(np.var(gd))

def loudness_lufs(y, sr):
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(y))

def analyze_audio(path):
    y, sr = load_audio(path)

    A = {
        "sample_rate": sr,
        "loudness_lufs": loudness_lufs(y, sr),
        "spectral_centroid": spectral_centroid(y, sr),
        "spectral_flatness": spectral_flatness(y),
        "spectral_flatness_high": spectral_flatness_high(y, sr),
        "rolloff_85": spectral_rolloff(y, sr, 0.85),
        "rolloff_95": spectral_rolloff(y, sr, 0.95),
        "high_freq_energy_10_16k": high_freq_energy(y, sr),
        "onset_strength": onset_strength(y, sr),
        "group_delay_variance": group_delay_variance(y, sr),
    }

    for key in A.keys():
        print(f"{key:25s} | A: {A[key]:12.4f}")


def analyze(audio_path):
    print("Loading audio...")
    y, sr = librosa.load(audio_path, sr=None)

    # Compute RMS for dBFS
    rms = librosa.feature.rms(y=y)[0]
    dB = librosa.amplitude_to_db(rms, ref=np.max)

    # Global loudness
    meter = Meter(sr)
    loudness_lufs = meter.integrated_loudness(y)

    # Pitch
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=350, sr=sr, frame_length=2048)
    avg_pitch = np.nanmean(f0)
    pitch_std = np.nanstd(f0)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = float(np.nanmean(centroid))

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    avg_flatness = float(np.mean(flatness))

    # Estimate formants via LPC (approx)
    def estimate_formants(frame, sr):
        A = librosa.lpc(frame, order=8)
        roots = np.roots(A)
        roots = roots[np.imag(roots) >= 0]
        angz = np.arctan2(np.imag(roots), np.real(roots))
        formants = sorted(angz * (sr / (2 * np.pi)))
        return formants[:2] if len(formants) >= 2 else [None, None]

    window = 2048
    f1_list, f2_list = [], []
    for i in range(0, len(y) - window, window):
        f1, f2 = estimate_formants(y[i:i + window], sr)
        if f1 and f2:
            f1_list.append(f1)
            f2_list.append(f2)

    avg_f1 = np.mean(f1_list)
    avg_f2 = np.mean(f2_list)

    # ------------------------------------------------------------
    # SILENCE DETECTION
    # ------------------------------------------------------------

    chunk = int(sr * (SILENCE_CHUNK_MS / 1000))
    frame_db_values = []

    # compute per-frame RMS in dB
    for i in range(0, len(y), chunk):
        frame = y[i:i + chunk]
        rms_val = np.sqrt(np.mean(frame ** 2) + 1e-12)
        db_val = librosa.amplitude_to_db(np.array([rms_val]), ref=np.max)[0]
        frame_db_values.append(db_val)

    frame_db_values = np.array(frame_db_values)

    # quick diagnostics (paste after frame_db_values computed)
    print("FRAME DB STATS: min, 10%, 25%, median, 75%, 90%, max")
    q = np.percentile(frame_db_values, [0, 10, 25, 50, 75, 90, 100])
    print(np.round(q, 2))
    # show how many frames fall say 4/6/8/10 dB below median
    med = np.median(frame_db_values)
    for d in (4, 6, 8, 10, 12):
        print(f"frames below median-{d}dB:", np.sum(frame_db_values < (med - d)))

    # Moving average
    def moving_average(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.concatenate([
            np.repeat(a[:n].mean(), n - 1),
            ret[n - 1:] / n
        ])

    ma = moving_average(frame_db_values, MA_WINDOW)

    # Silence = current frame is lower than moving average minus RELATIVE_DIP_DB
    silence_mask = frame_db_values < (ma - RELATIVE_DIP_DB)

    # group segments
    silent_segments = []
    start = None
    for idx, is_silent in enumerate(silence_mask):
        if is_silent and start is None:
            start = idx
        if not is_silent and start is not None:
            silent_segments.append((start, idx))
            start = None
    if start is not None:
        silent_segments.append((start, len(silence_mask)))

    print("Detected silent segments:")
    for s, e in silent_segments:
        dur = (e - s) * SILENCE_CHUNK_MS
        if dur >= MIN_SILENCE_MS:
            print(f"  {s * SILENCE_CHUNK_MS / 1000:.2f}s → {e * SILENCE_CHUNK_MS / 1000:.2f}s ({dur:.1f}ms)")

    # # ------------------------------------------------------------
    # # PLOTS
    # # ------------------------------------------------------------
    fig, ax = plt.subplots(4, 1, figsize=(14, 10))


    # # Waveform + silence mask
    # times = np.arange(len(y)) / sr
    # ax[0].plot(times, y, linewidth=0.7)
    # ax[0].set_title("Waveform with Silence Regions Highlighted")
    #
    # # overlay silent chunks
    # mask_times = np.arange(len(silence_mask)) * (SILENCE_CHUNK_MS / 1000)
    # silent_points = np.array(silence_mask, dtype=int)
    # ax[0].scatter(mask_times, silent_points * np.max(y), color="red", s=8, label="silence mask")
    #
    # # highlight long silent segments
    # for s, e in silent_segments:
    #     dur = (e - s) / sr * 1000
    #     if dur > MIN_SILENCE_MS:
    #         ax[0].axvspan(s / sr, e / sr, color="yellow", alpha=0.3)

    # ax[0].legend()
    #
    # # Pitch
    # ax[1].set_title("Pitch (F0) Curve")
    # ax[1].plot(f0)
    #
    # # Spectral centroid
    # ax[2].set_title("Brightness (Spectral Centroid)")
    # ax[2].plot(centroid)
    #
    # # RMS
    # ax[3].set_title("RMS Loudness")
    # ax[3].plot(rms)
    #
    # plt.tight_layout()
    # plt.show()

    # ------------------------------------------------------------
    # PRINT SUMMARY METRICS
    # ------------------------------------------------------------
    print("\n==== TONE ANALYSIS ====")
    print(f"Loudness (LUFS):       {loudness_lufs:.2f}")
    print(f"Pitch (mean F0 Hz):     {avg_pitch:.2f}")
    print(f"Pitch drift (std Hz):   {pitch_std:.2f}")
    print(f"Brightness centroid:    {avg_centroid:.2f}")
    print(f"Timbre flatness:        {avg_flatness:.4f}")
    print(f"Formant F1 (warmth):    {avg_f1:.2f}")
    print(f"Formant F2 (nasality):  {avg_f2:.2f}")


if __name__ == "__main__":

    audio = "output/audios/Blake (Chapter 30)/Blake (Chapter 30).m4a"

    analyze(audio)

    analyze_audio(audio)
