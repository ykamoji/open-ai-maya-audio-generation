import torch
import os
import numpy as np
from pathlib import Path
from Emotions.utils import clear_cache

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


def learn_silence_frame(snac_model):


    # generate 2 seconds of PCM silence
    sil_pcm = torch.zeros(1, 1, SAMPLING_RATE * 2, dtype=torch.float32, device='cuda:0')

    with torch.inference_mode():
        L1, L2, L3 = snac_model.encode(sil_pcm)

    L1 = L1[0].cpu().numpy()
    L2 = L2[0].cpu().numpy()
    L3 = L3[0].cpu().numpy()

    mid = L1.shape[0] // 2

    # stable silence clusters
    L1_s = int(L1[mid])
    L2a_s = int(L2[mid, 0])
    L2b_s = int(L2[mid, 1])
    L3a_s = int(L3[mid, 0])
    L3b_s = int(L3[mid, 1])
    L3c_s = int(L3[mid, 2])
    L3d_s = int(L3[mid, 3])

    silence_frame = [
        CODE_TOKEN_OFFSET + L1_s,
        CODE_TOKEN_OFFSET + L2a_s,
        CODE_TOKEN_OFFSET + L3a_s,
        CODE_TOKEN_OFFSET + L3b_s,
        CODE_TOKEN_OFFSET + L2b_s,
        CODE_TOKEN_OFFSET + L3c_s,
        CODE_TOKEN_OFFSET + L3d_s,
    ]

    return silence_frame


def generate_true_silence(duration_sec, silence_frame, samples_per_frame):
    frame_duration = samples_per_frame / SAMPLING_RATE
    frames = int(round(duration_sec / frame_duration))
    return silence_frame * frames


def saveAudio(outputPath, audio_frames, title):
    file = os.path.join(outputPath, f"{title}.npy")
    np.save(file, audio_frames)
    # print(f"Saved {title}.")
    # sf.write(outputPath + f'{title}.wav', audio_frames, samplerate=24000)


def decode_audio(generated_snac_tokens, snac_model):
    # Extract SNAC audio Level tokens
    levels = unpack_snac_from_7(generated_snac_tokens)

    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device='cuda:0').unsqueeze(0)
        for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].float().cpu().numpy()

    del codes_tensor
    del z_q
    clear_cache()

    return audio.astype(np.float32)


def generate_snac_silence_tokens(duration_sec):
    """
    SNAC silence = using RVQ index 0 at all levels.
    Therefore token_id = CODE_TOKEN_OFFSET.
    """
    if duration_sec <= 0:
        return []

    frame_duration = SAMPLES_PER_FRAME / float(SAMPLING_RATE)
    frames = int(round(duration_sec / frame_duration))

    return [CODE_TOKEN_OFFSET] * (frames * SNAC_TOKENS_PER_FRAME)


def measure_samples_per_frame(snac_model, silence_frame):
    one_frame_audio = decode_audio(silence_frame, snac_model)
    return len(one_frame_audio)

# --- Main assembler ---
def assemble_snac_segments_and_stitch(
        generated_parts_tokens,        # list of token sequences per chunk
        snac_model,
        silence_frame,
        samples_per_frame,
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
    snac_codes_per_chunk = [extract_snac_codes(toks) for toks in generated_parts_tokens]

    final_codes = []

    for i, part_codes in enumerate(snac_codes_per_chunk):
        if part_codes:
            final_codes.extend(part_codes)

        # Add codec-consistent silence frames
        if i < num_chunks - 1:
            silence_tokens = generate_true_silence(silence_schedule[i], silence_frame, samples_per_frame)
            final_codes.extend(silence_tokens)

    # ------------------------------
    # 3. Decode all in one go
    # ------------------------------
    final_audio = decode_audio(final_codes, snac_model)
    return final_audio.reshape(-1)


def create_audio(generated_outputs, snac_model, audio_path, para_breaks, tagged_list, title):
    completed = True
    try:

        data = {
            "chunks": generated_outputs,
            "para_breaks": para_breaks,
            "tagged_list": tagged_list,
        }

        Path(audio_path).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(audio_path, f"{title}_meta.npy"), data)

        # learn silence frame
        silence_frame = learn_silence_frame(snac_model)

        # true frame size
        samples_per_frame = measure_samples_per_frame(snac_model, silence_frame)

        stream = []
        for gen_out in generated_outputs:
            stream.extend(extract_snac_codes(gen_out))
        audio_frames = decode_audio(stream, snac_model)
        saveAudio(audio_path, audio_frames, title+'_normal')
        audio_frames = assemble_snac_segments_and_stitch(generated_outputs, snac_model, silence_frame, samples_per_frame, para_breaks, tagged_list)
        saveAudio(audio_path, audio_frames, title+'_stitch')
    except Exception as e:
        print(f"Snac model Decoding error: {e}")
        completed = False

    return completed


if __name__ == '__main__':
    audio_path = ""
    title = ""
    data = np.load(os.path.join(audio_path, f"{title}_meta.npy"), allow_pickle=True).item()
    generated_outputs = data["chunks"]
    para_breaks = data["para_breaks"]
    tagged_list = data["tagged_list"]

    create_audio(generated_outputs, snac_model=None, audio_path=audio_path, para_breaks=para_breaks, tagged_list=tagged_list)
