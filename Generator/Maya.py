import torch
import os
import glob
import time
import warnings
import random
import numpy as np
import threading
import queue
import soundfile as sf
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Emotions.utils import clear_cache, getDevice
from Generator.utils import batch_sentences
from snac import SNAC

warnings.filterwarnings("ignore")

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009

LOG_STEPS = 1

SAMPLE_RATE = 24000
HOP_SAMPLES = 320
SNAC_BATCH_SUBBATCH = 8
CROSSFADE_MS = 20
SILENCE_MS = 30


def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token

    formatted_text = f'<description="{description}"> {text}'

    prompt = (
            soh_token + bos_token + formatted_text + eot_token +
            eoh_token + soa_token + sos_token
    )

    return prompt


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


def batch_decode_snac_levels(levels_lists, snac_model, device):
    audios = []

    for levels in levels_lists:
        codes_tensor = [
            torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
            for level in levels
        ]

        with torch.inference_mode():
            z_q = snac_model.quantizer.from_codes(codes_tensor)
            audios.append(snac_model.decoder(z_q)[0, 0].cpu().numpy())

    return audios


def make_silence(ms, sr=SAMPLE_RATE):
    samples = int(sr * (ms / 1000.0))
    return np.zeros(samples, dtype=np.float32)


def crossfade(a, b, fade_ms=CROSSFADE_MS, sr=SAMPLE_RATE):
    if fade_ms <= 0:
        return np.concatenate([a, b])
    fade_samples = int(sr * (fade_ms / 1000.0))
    if len(a) < fade_samples or len(b) < fade_samples:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    a_tail = a[-fade_samples:] * fade_out
    b_head = b[:fade_samples] * fade_in
    mid = a_tail + b_head
    return np.concatenate([a[:-fade_samples], mid, b[fade_samples:]])


def stitch_audio_list(audio_list, silence_ms=SILENCE_MS, crossfade_ms=CROSSFADE_MS):
    if not audio_list:
        return np.zeros(1, dtype=np.float32)
    out = audio_list[0]
    for next_audio in audio_list[1:]:
        if crossfade_ms > 0:
            out = crossfade(out, next_audio, fade_ms=crossfade_ms)
        else:
            silence = make_silence(silence_ms)
            out = np.concatenate([out, silence, next_audio])
    return out


def getDescription(MayaArgs, title):
    description = ""
    for character in MayaArgs.Characters:
        if character.Name in title:
            description = character.Description
    print(f"Description: {description}")
    return description


def estimate_audio_duration(generated_ids):
    snac_count = len([SNAC_MIN_ID <= t <= SNAC_MAX_ID for t in generated_ids])
    duration_sec = snac_count * 0.01194 + 0.07499
    return duration_sec


def getModels(MODEL_NAME, CACHE_PATH):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME if platform != "Kaggle" else MODEL_NAME + 'Model/',
        cache_dir=CACHE_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = model.eval()

    model.generation_config.cache_implementation = "static"

    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz",
                                      cache_dir=CACHE_PATH).eval()
    print("Models loaded")
    return model, snac_model, tokenizer


def getTokenizer(MODEL_NAME, CACHE_PATH):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME if platform != "Kaggle" else MODEL_NAME + 'Tokenizer/',
        cache_dir=CACHE_PATH,
        trust_remote_code=True
    )
    return tokenizer


def delete_previous_outputs(outputPath, step):
    files = glob.glob(outputPath + "/partial_*.wav")
    files.sort(key=os.path.getmtime)
    files_to_delete = files[:-2]
    if files_to_delete and step > 0 and step % LOG_STEPS == 0:
        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception as e:
                pass


def decode_audio(outputs, snac_model):
    snac_lists = [extract_snac_codes(ids) for ids in outputs]
    levels_lists = [unpack_snac_from_7(snac) for snac in snac_lists]
    audio_chunks = batch_decode_snac_levels(levels_lists, snac_model, getDevice())
    return audio_chunks


def convert(Args, content, title, outputPath):
    torch.set_grad_enabled(False)
    MayaArgs = Args.Generator.Maya

    global platform
    platform = Args.Platform

    MODEL_NAME = MayaArgs.ModelName.__dict__[Args.Platform]
    CACHE_PATH = MayaArgs.CachePath.__dict__[Args.Platform]

    description = getDescription(MayaArgs, title)

    GPUCount = Args.Generator.GPU.__dict__[Args.Platform]

    chunks = batch_sentences(content)

    model, snac_model, tokenizer = getModels(MODEL_NAME, CACHE_PATH)

    tokenized_inputs = [
        tokenizer(build_prompt(tokenizer, description, chunk), return_tensors="pt")
        if chunk.strip() else ""
        for chunk in chunks
    ]

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if GPUCount > 0:
        print("Running in GPU env.")
        torch.cuda.manual_seed_all(0)
        # multiGPU(chunks, description, outputPath, title, MODEL_NAME, CACHE_PATH)
    else:
        print("Running in CPU env.")
        cpuProcess(tokenized_inputs, model, outputPath, snac_model, title, tokenizer.pad_token_id)


def processVoice(model, inputs, pad_token_id):
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs.to(model.device),
            max_new_tokens=2048,
            min_new_tokens=28,  # At least 4 SNAC frames
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,  # Prevent loops
            eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
            do_sample=True,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
    generation_time = time.time() - start_time
    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
    audio_duration = estimate_audio_duration(generated_ids)
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

    del outputs
    clear_cache()

    return generated_ids, generation_time, audio_duration, rtf


def saveAudio(outputPath, audio_chunks, title):
    silence_first = np.zeros(int(0.15 * 24000), dtype=np.float32)
    silence_normal = np.zeros(int(0.1 * 24000), dtype=np.float32)
    full = []
    for i, chunk in enumerate(audio_chunks):
        if chunk is None or len(chunk) == 0:
            continue
        full.append(chunk)
        if i < len(audio_chunks) - 1:
            if i == 0:
                full.append(silence_first)
            else:
                full.append(silence_normal)

    if not full:
        full_audio = np.zeros(1, dtype=np.float32)
    else:
        full_audio = np.concatenate(full)
    file = os.path.join(outputPath, f"{title}.npy")
    Path(outputPath).mkdir(parents=True, exist_ok=True)
    # np.save(file, full_audio)
    sf.write(os.path.join(outputPath, f"{title}.wav"), full_audio, 24000)


def cpuProcess(prompt_inputs, model, outputPath, snac_model, title, pad_token_id):
    audio_chunks = []
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    model = model.to(getDevice())
    snac_model = snac_model.to(getDevice())
    audio_path = outputPath + f"audios/{title}/"
    outputs = []
    for part, input in enumerate(tqdm(prompt_inputs, desc=f"{title}")):
        if type(input) == str:
            # Adding a pause between paras to keep the conversation seperate
            audio_chunks.append(np.zeros(int(0.15 * 24000)))
            # print(f"Voice generation for part {step} (para break)")
            continue
        # print(f"Voice generation for part {step} ...")
        generated_tokens, generation_time, audio_duration, rtf = processVoice(model, input, pad_token_id)
        outputs.append(generated_tokens)
        # audio_duration = (len(audio) / 24000)
        print(f"Voice generation for part {step} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
        # audio_chunks.append(audio)
        input_length = len(input['input_ids'][0])
        input_lengths.append(input_length)
        generation_times.append(generation_time)

        if step % LOG_STEPS == 0:
            writer.add_scalar("Evaluation/InputSize", input_length, step)
            writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
            writer.add_scalar("Performance/GenerationTime", generation_time, step)
            writer.add_scalar("Performance/RTF", rtf, step)

            if step > 2:
                correlation = np.corrcoef(input_lengths, generation_times)[0, 1]
                writer.add_scalar("Performance/InputDurationCorr", correlation, step)

            delete_previous_outputs(audio_path, step)
            audio_chunks = decode_audio(outputs, snac_model)
            saveAudio(audio_path, audio_chunks, f"partial_{step}")

        step += 1
    writer.close()

    audio_chunks = decode_audio(outputs, snac_model)

    saveAudio(audio_path, audio_chunks, title)

#
# def multiGPU(chunks, description, outputPath, title, MODEL_NAME, CACHE_PATH):
#     writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
#     available_gpus = torch.cuda.device_count()
#
#     q = queue.Queue()
#
#     count = 0
#     for idx, text in enumerate(chunks):
#         q.put((idx, text.strip()))
#         count += 1
#
#     print(f"Total prompts {count}")
#
#     lock = threading.Lock()
#     threads = []
#
#     sharedData = {
#         "results": {},
#         "global_step": 0,
#         "input_lengths": [],
#         "generation_times": []
#     }
#
#     pbar = tqdm(total=len(chunks), desc="Generating audio")
#
#     models = []
#     snac_models = []
#     for i in range(available_gpus):
#         with torch.device(f"cuda:{i}"):
#             model, snac_model, tokenizer = getModels(MODEL_NAME, CACHE_PATH)
#             models.append(model)
#             snac_models.append(snac_model)
#
#     # Voice warmup
#     for model in models:
#         dummy = torch.randint(0, 30000, (1, 7), device=model.device, dtype=torch.long)
#         with torch.inference_mode():
#             _ = model.model.layers[0](model.model.embed_tokens(dummy))
#
#     audio_path = outputPath + f"audios/{title}/"
#
#     for gpu_id in range(available_gpus):
#         t = threading.Thread(target=gpu_worker,
#                              args=(f"cuda:{gpu_id}", q, models[gpu_id], snac_models[gpu_id], tokenizer, description,
#                                    sharedData, lock, writer, audio_path, pbar))
#         t.start()
#         threads.append(t)
#
#     q.join()
#
#     for t in threads:
#         t.join()
#
#     writer.flush()
#     writer.close()
#
#     ordered_audios = sharedData["results"]
#     ordered_indices = sorted(ordered_audios.keys())
#     full_audio = [ordered_audios[idx] for idx in ordered_indices]
#     saveAudio(audio_path, full_audio, title)
#
#
# def gpu_worker(gpu_id, q, model, snac_model, tokenizer, description, sharedData, lock, writer, outputPath, pbar):
#     device = torch.device(gpu_id)
#     while True:
#         try:
#             idx, text = q.get(timeout=2)
#         except queue.Empty:
#             print(f"[{gpu_id}] No more prompts. Shutting down.")
#             break
#
#         set_seed(base_seed=4321, chunk_idx=idx, gpu_id=gpu_id)
#
#         if not text.strip():
#             audio = np.zeros(int(0.2 * 24000))
#             generation_time = 0
#             audio_duration = 0
#             # print(f"[{gpu_id}] Voice generation for part {idx} (para break)")
#         else:
#             # print(f"[{gpu_id}] Voice generation for part {idx} ...")
#             start_time = time.time()
#             audio = processVoice(model, device, tokenizer, snac_model, text, description, idx)
#             generation_time = time.time() - start_time
#             audio_duration = (len(audio) / 24000)
#             # print(f"[{gpu_id}] Voice generation for part {idx} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
#
#         with lock:
#             sharedData["results"][idx] = audio
#             if text.strip():
#                 pbar.update(1)
#                 text_len = len(text)
#                 sharedData["global_step"] += 1
#                 step = sharedData["global_step"]
#                 sharedData["input_lengths"].append(text_len)
#                 sharedData["generation_times"].append(generation_time)
#                 if step > 1 and step % LOG_STEPS == 0:
#                     writer.add_scalar("Evaluation/InputSize", text_len, step)
#                     writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
#                     writer.add_scalar("Performance/GenerationTime", generation_time, step)
#
#                     rtf = (generation_time / audio_duration) if audio_duration > 0 else float('inf')
#                     writer.add_scalar("Performance/RTF", rtf, step)
#
#                     correlation = np.corrcoef(sharedData["input_lengths"], sharedData["generation_times"])[0, 1]
#                     writer.add_scalar("Performance/InputDurationCorr", correlation, step)
#
#                     if step % (LOG_STEPS * 2) == 0:
#                         # Save partial audios
#                         partial_audios = sharedData["results"]
#                         partial_indices = sorted(partial_audios.keys())
#                         partial_audio = [partial_audios[idx] for idx in partial_indices]
#
#                         delete_previous_outputs(outputPath, step)
#                         saveAudio(outputPath, partial_audio, f"partial_{step}")
#
#         q.task_done()
#         torch.cuda.empty_cache()