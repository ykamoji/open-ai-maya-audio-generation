import torch
import os
import numpy as np
from pathlib import Path
from Emotions.utils import getDevice, clear_cache

CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SAMPLING_RATE = 24000
DECODER_RATES = [8, 8, 4, 2]
SAMPLES_PER_FRAME = int(np.prod(DECODER_RATES))


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


def saveAudio(outputPath, audio_frames, title):
    file = os.path.join(outputPath, f"{title}.npy")
    Path(outputPath).mkdir(parents=True, exist_ok=True)
    np.save(file, audio_frames)
    # print(f"Saved {title}.")
    # sf.write(outputPath + f'{title}.wav', audio_frames, samplerate=24000)


def decode_audio(generated_snac_tokens, snac_model):

    # Extract SNAC audio Level tokens
    levels = unpack_snac_from_7(generated_snac_tokens)

    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=getDevice()).unsqueeze(0)
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
        snac_model,
        para_break_indices=None,       # NEW: list of chunk indices where paragraph breaks occur
        tagged_list=None,               # NEW: list[bool], same length as generated_parts_tokens
        silence_between_chunks=0.25,   # default normal silence
        tagged_silence=0.30,           # silence after tagged lines
        paragraph_silence=0.50,        # silence after paragraph break
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
    snac_codes_per_chunk = []
    frames_per_chunk = []

    for tokens in generated_parts_tokens:
        codes = extract_snac_codes(tokens)
        snac_codes_per_chunk.append(np.array(codes, dtype=np.int64))
        frames = len(codes) // SNAC_TOKENS_PER_FRAME
        frames_per_chunk.append(frames)

    total_frames = sum(frames_per_chunk)
    if total_frames == 0:
        return np.array([], dtype=np.float32)

    concatenated_codes = np.concatenate(snac_codes_per_chunk, axis=0).tolist()

    # ------------------------------
    # 3. Decode once
    # ------------------------------
    audio = decode_audio(concatenated_codes, snac_model)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    # ------------------------------
    # 4. Split decoded waveform back into chunk segments
    # ------------------------------
    segments = []
    cursor = 0

    for frames in frames_per_chunk:
        samples = frames * SAMPLES_PER_FRAME
        seg = audio[cursor : cursor + samples]
        segments.append(seg.copy())
        cursor += samples

    if cursor < len(audio):
        segments[-1] = np.concatenate([segments[-1], audio[cursor:]], axis=0)

    # ------------------------------
    # 5. Stitch with silence
    # ------------------------------
    final_parts = []

    for i, seg in enumerate(segments):
        seg = np.asarray(seg, dtype=np.float32)
        final_parts.append(seg)

        if i < len(segments) - 1:
            silence_sec = silence_schedule[i]
            silence_samples = int(round(silence_sec * SAMPLING_RATE))
            silence = np.zeros(silence_samples, dtype=np.float32)
            final_parts.append(silence)

    final_audio = np.concatenate(final_parts, axis=0)
    return final_audio


def create_audio(generated_outputs, snac_model, audio_path, para_breaks, tagged_list, title):
    completed = True
    try:
        # audio_frames = decode_audio(generated_outputs, snac_model)
        audio_frames = assemble_snac_segments_and_stitch(generated_outputs, snac_model, para_breaks, tagged_list)
        saveAudio(audio_path, audio_frames, title)
    except Exception as e:
        print(f"Snac model Decoding error: {e}")
        completed = False

    return completed
