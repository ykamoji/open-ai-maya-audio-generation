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

    levels = unpack_snac_from_7(generated_snac_tokens)

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

    return audio.astype(np.float32)


def create_audio(generated_outputs, snac_model, audio_path, title):

    audio_frames = decode_audio(generated_outputs, snac_model)

    saveAudio(audio_path, audio_frames, title)

