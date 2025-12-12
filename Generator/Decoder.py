import torch
import os
import re
import json
import srt
from datetime import timedelta
import sys
import numpy as np
import glob
import argparse
from tqdm import tqdm
from pathlib import Path
from Emotions.utils import clear_cache, getDevice
from Generator.utils import getSnacModel, batch_sentences
from PostProcess.auto_tone_equalize import process_npy

CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SAMPLING_RATE = 24000
SAMPLES_PER_FRAME = 2048


def extract_snac_codes(token_ids):
    """Extract SNAC codes from generated tokens."""
    try:
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids)

        eos_positions = np.where(token_ids == CODE_END_TOKEN_ID)[0]
        eos_idx = eos_positions[0] if len(eos_positions) > 0 else len(token_ids)
    except ValueError:
        eos_idx = len(token_ids)

    snac_mask = (token_ids >= SNAC_MIN_ID) & (token_ids <= SNAC_MAX_ID)
    return token_ids[:eos_idx][snac_mask[:eos_idx]].tolist()


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


def generate_true_silence(duration_sec, silence_frame, samples_per_frame):
    frame_duration = samples_per_frame / SAMPLING_RATE
    frames = int(round(duration_sec / frame_duration))
    return silence_frame * frames


def saveAudio(outputPath, audio_frames, title):
    file = os.path.join(outputPath, f"{title}.npy")
    np.save(file, audio_frames)
    # print(f"Saved {title}.")
    # sf.write(outputPath + f'{title}.wav', audio_frames, samplerate=24000)


def decode_audio(generated_snac_tokens, snac_model, device):
    # Extract SNAC audio Level tokens
    levels = unpack_snac_from_7(generated_snac_tokens)

    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0) # 'cuda:0'
        for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].float().cpu().numpy()

    del codes_tensor
    del z_q
    clear_cache()

    return audio.astype(np.float32)


# --- Main assembler ---
def assemble_snac_segments_and_stitch(
        generated_parts_tokens,        # list of token sequences per chunk
        silence_frame,
        samples_per_frame,
        para_break_indices=None,       # NEW: list of chunk indices where paragraph breaks occur
        tagged_list=None,               # NEW: list[bool], same length as generated_parts_tokens
        silence_between_chunks=0.20,   # default normal silence
        tagged_silence=0.32,           # silence after tagged lines
        paragraph_silence=0.60,        # silence after paragraph break
):
    """
    Assemble SNAC segments and add silence between chunks with paragraph logic.
    """

    num_chunks = len(generated_parts_tokens)

    # ------------------------------
    # 1. Build silence schedule
    # ------------------------------
    silence_schedule = []

    # Build schedule automatically using:
    #  - is_tagged_list (optional)
    #  - para_break_indices (optional)
    para_break_indices = set(para_break_indices or [])
    tagged_list = tagged_list or [False] * num_chunks

    for i in range(num_chunks - 1):
        if i in para_break_indices:
            silence_schedule.append(paragraph_silence)  # 500 ms
        elif tagged_list[i]:
            silence_schedule.append(tagged_silence)  # 300 ms
        else:
            silence_schedule.append(silence_between_chunks)  # 200–250 ms

    # ------------------------------
    # 2. Extract SNAC codes per chunk
    # ------------------------------
    snac_codes_per_chunk = [extract_snac_codes(toks) for toks in generated_parts_tokens]

    final_codes = []

    for i, part_codes in enumerate(snac_codes_per_chunk):
        if part_codes:
            final_codes.append(("speech", part_codes))

        # Add codec-consistent silence frames
        if i < num_chunks - 1:
            silence_tokens = generate_true_silence(silence_schedule[i], silence_frame, samples_per_frame)
            final_codes.append(("silence", silence_tokens))

    return final_codes


def save_snac_tokens(generated_outputs, audio_path, para_breaks, tagged_list, title):
    completed = True
    try:
        data = {
            "chunks": generated_outputs,
            "para_breaks": para_breaks,
            "tagged_list": tagged_list,
        }
        Path(audio_path).mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(os.path.join(audio_path, f"{title}_meta.npy")):
            np.save(os.path.join(audio_path, f"{title}_meta.npy"), data)
        else:
            nxt = 1
            while True:
                if not os.path.isfile(os.path.join(audio_path, f"{title}_{nxt}_meta.npy")):
                    np.save(os.path.join(audio_path, f"{title}_meta_{nxt}.npy"), data)
                    break
                nxt += 1

    except Exception as e:
        print(f"Saving snac tokens meta data error: {e}")
        completed = False

    return completed


def getDialogues(title):

    with open("cache/emotionCache.json") as f:
        data = json.load(f)

    lines = []
    for notebook in data:
        for section in data[notebook]:
            for page in data[notebook][section]:
                if title in page:
                    lines = data[notebook][section][page]
                    break

    audio_lines, _, _, _ = batch_sentences(lines)
    return audio_lines


def write_srt(sentences, timeline, out_path):

    subtitles = []
    for idx, ((start, end), text) in enumerate(zip(timeline, sentences), start=1):
        subtitles.append(
            srt.Subtitle(
                index=idx,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text.strip()
            )
        )

    srt_output = srt.compose(subtitles)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_output)


def decode(device, generated_outputs, snac_model, audio_path, para_breaks, tagged_list, title):
    completed = True
    try:
        # learn silence frame
        silence_frame = [130334, 130002, 131688, 131688, 130002, 131688, 131688]

        # true frame size
        samples_per_frame = 2048

        # speed up
        speed_factor = 1.25

        # audio_frames = []
        # for gen_out in tqdm(generated_outputs, desc=f"Normal", ncols=100, file=sys.stdout):
        #     audio_frames.extend(decode_audio(extract_snac_codes(gen_out), snac_model, device))
        #
        # saveAudio(audio_path, audio_frames, title + '_n')
        # process_npy(input_path=os.path.join(audio_path, f"{title + '_n'}.npy"),
        #             output_wav=os.path.join(audio_path, f"audiobook_{getChapter(title)}_n.npy"))

        lines = getDialogues(title)
        silence_between_chunks = 0.20 / speed_factor
        tagged_silence = 0.30 / speed_factor
        paragraph_silence = 0.60 / speed_factor

        final_codes = assemble_snac_segments_and_stitch(generated_outputs,
                                                        silence_frame,
                                                        samples_per_frame,
                                                        para_breaks,
                                                        tagged_list,
                                                        silence_between_chunks,
                                                        tagged_silence,
                                                        paragraph_silence)
        audio_frames = []
        saved_decoded_audio = []
        decoded_audio_available = True
        if os.path.isfile(os.path.join(audio_path, f"{title}_decoded.npy")):
            saved_decoded_audio = np.load(os.path.join(audio_path, f"{title}_decoded.npy"), allow_pickle=True).tolist()
        else:
            decoded_audio_available = False

        timeline = []
        t = 0.0
        i = 0
        for (type, gen_out) in tqdm(final_codes, desc=f"{title}", ncols=100, file=sys.stdout):
            if decoded_audio_available:
                decoded = saved_decoded_audio[i]
            else:
                decoded = decode_audio(gen_out, snac_model, device)
            i += 1

            duration = len(decoded) / SAMPLING_RATE

            if type == 'speech':
                start = t
                end = start + duration
                timeline.append((start, end))

            t += duration

            audio_frames.extend(decoded)
            if not decoded_audio_available:
                saved_decoded_audio.append(decoded)

        if not decoded_audio_available:
            np.save(os.path.join(audio_path, f"{title}_decoded.npy"), np.array(saved_decoded_audio, dtype=object))

        saveAudio(audio_path, audio_frames, title)

        process_npy(input_path=os.path.join(audio_path, f"{title}.npy"),
                    output_wav=os.path.join(audio_path, f"{title}.wav"),
                    tempo=speed_factor)

        timeline = [(s / speed_factor, e / speed_factor) for (s, e) in timeline]

        write_srt(lines, timeline, os.path.join(audio_path, f"{title}.srt"))

        try:
            os.remove(os.path.join(audio_path, f"{title}.npy"))
        except Exception:
            pass

    except Exception as e:
        print(f"Decoding error: {e}")
        completed = False

    return completed


def getChapterNo(title):
    return int(re.search(r'\d+', title).group())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Decoding")
    parser.add_argument("--path", type=str, default="/output", help="Audio Output Path")
    parser.add_argument("--modelPath", type=str, default="/models", help="Snac Model Path")
    parser.add_argument("--limits", type=json.loads, default=None, help="Range")
    args = parser.parse_args()

    path = args.path
    model_path = args.modelPath
    limits = args.limits

    if not os.path.isdir(path):
        raise Exception(f"{path} is not a directory. Check and give correct path.")

    if not os.path.isdir(model_path):
        raise Exception(f"{model_path} is not a directory. Check and give correct path.")

    files = [file for file in glob.glob(os.path.join(path, f"audios/*/*.npy")) if "_meta" in file]

    snac_model = getSnacModel(model_path)
    device = 'cuda:0' if torch.cuda.is_available() else getDevice()
    snac_model.to(device)

    files.sort(key=lambda x: getChapterNo(x))
    start = 0
    end = len(files)
    if limits:
        start = limits[0]
        end = limits[1]

    for file in files:
        currentChapter = getChapterNo(file)
        if not start <= currentChapter <= end:
            continue

        title = os.path.split(file)[-1].split("_")[0]
        audio_path = os.path.split(file)[-2]
        data = np.load(file, allow_pickle=True)
        data = data.item()
        generated_outputs = data["chunks"]
        para_breaks = data["para_breaks"]
        tagged_list = data["tagged_list"]
        completed = decode(device=device,
                           generated_outputs=generated_outputs,
                           snac_model=snac_model,
                           audio_path=audio_path,
                           para_breaks=para_breaks,
                           tagged_list=tagged_list,
                           title=title)
        if completed:
            print(f"Decoding completed for {title} !")

        else:
            print(f"Error for {title}. Check error logs.")

    ## TO check part voices
    # gen_out = np.load("output/audios/part_5.npy")
    #
    # audio = decode_audio(extract_snac_codes(gen_out), snac_model, device)
    # import soundfile as sf
    # sf.write("output/audios/check.wav", audio, SAMPLING_RATE, "FLOAT")


