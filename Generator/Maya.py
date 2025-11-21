import torch
import os
import glob
import time
import warnings
import numpy as np
import threading
import queue
import gc
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
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

LOG_STEPS = 20


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


def getDescription(MayaArgs, title):
    description = ""
    for character in MayaArgs.Characters:
        if character.Name in title:
            description = character.Description
    print(f"Description: {description}")
    return description


def getModels(MODEL_PATH):
    model = AutoModelForCausalLM.from_pretrained(
        "maya-research/maya1" if platform != "Kaggle" else '/kaggle/input/maya/transformers/1/2/Model',
        cache_dir=MODEL_PATH,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    tokenizer = getTokenizer(MODEL_PATH)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz",  cache_dir=MODEL_PATH).eval()
    print("Models loaded")
    return model, snac_model, tokenizer


def getTokenizer(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(
        "maya-research/maya1" if platform != "Kaggle" else '/kaggle/input/maya/transformers/1/2/Tokenizer',
        cache_dir=MODEL_PATH,
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


def convert(Args, content, title, outputPath):
    MayaArgs = Args.Generator.Maya

    global platform
    platform = Args.Platform

    MODEL_PATH = MayaArgs.ModelPath.__dict__[Args.Platform]

    description = getDescription(MayaArgs, title)

    GPUCount = Args.Generator.GPU.__dict__[Args.Platform]

    chunks = batch_sentences(content)

    torch.manual_seed(3542)

    if GPUCount > 0:
        print("Running in GPU env.")
        multiGPU(chunks, description, outputPath, title, MODEL_PATH)
    else:
        print("Running in CPU env.")
        model, snac_model, tokenizer = getModels(MODEL_PATH)
        cpuProcess(chunks, description, model, outputPath, snac_model, title, tokenizer)


def processVoice(model, device, tokenizer, snac_model, text, description, part):
    prompt = build_prompt(tokenizer, description, text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    global generated_ids
    EOS_FOUND = False
    max_new_tokens = 2048
    while not EOS_FOUND:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # Increase to let model finish naturally
                min_new_tokens=28,  # At least 4 SNAC frames
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,  # Prevent loops
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
                # pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                # attn_implementation="flash_attention_2"
            )

            generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()

        if CODE_END_TOKEN_ID in generated_ids:
            EOS_FOUND = True
            # eos_position = generated_ids.index(CODE_END_TOKEN_ID)
            # print(f"Part {part} EOS token found at position {eos_position}/{len(generated_ids)}")
        else:
            EOS_FOUND = False
            generated_ids = None
            max_new_tokens *= 2
            print(f"Part {part} {text[:50]}... EOS token not found. Running again with max_token {max_new_tokens}")

    # Extract SNAC audio tokens
    snac_tokens = extract_snac_codes(generated_ids)

    if len(snac_tokens) < 7:
        print(f"Part {part} Error: Not enough SNAC tokens generated")
        # return

    levels = unpack_snac_from_7(snac_tokens)

    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()

    del z_q, codes_tensor, generated_ids, inputs
    torch.mps.empty_cache()
    gc.collect()

    return audio


def saveAudio(outputPath, audio_chunks, title):
    silence_first = np.zeros(int(0.3 * 24000), dtype=np.float32)
    silence_normal = np.zeros(int(0.15 * 24000), dtype=np.float32)
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
    np.save(file, full_audio)


def cpuProcess(chunks, description, model, outputPath, snac_model, title, tokenizer):
    audio_chunks = []
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    device = "mps" if not torch.backends.mps.is_built() and torch.backends.mps.is_available() else "cpu"
    if device == "mps":
        model = model.to(device)
        snac_model = snac_model.to(device)
    audio_path = outputPath + f"audios/{title}/"
    for part, chunk in enumerate(tqdm(chunks, desc="Generating audio")):
        # print(chunk)
        if chunk.strip():
            # Adding a pause between paras to keep the conversation seperate
            audio_chunks.append(np.zeros(int(0.2 * 24000)))
            # print(f"Voice generation for part {step} (para break)")
            continue
        # print(f"Voice generation for part {step}/{total} ...")
        start_time = time.time()
        audio = processVoice(model, device, tokenizer, snac_model, chunk, description, part)
        generation_time = time.time() - start_time
        audio_duration = (len(audio) / 24000)
        # print(f"Voice generation for part {step}/{total} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
        audio_chunks.append(audio)
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        input_length = len(chunk)
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
            saveAudio(audio_path, audio_chunks, f"partial_{step}")

        step += 1
    writer.close()
    saveAudio(audio_path, audio_chunks, title)


def multiGPU(chunks, description, outputPath, title, MODEL_PATH):
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    available_gpus = torch.cuda.device_count()

    q = queue.Queue()

    count = 0
    for idx, text in enumerate(chunks):
        q.put((idx, text.strip()))
        count += 1

    print(f"Total prompts {count}")

    lock = threading.Lock()
    threads = []

    sharedData = {
        "results": {},
        "global_step": 0,
        "input_lengths": [],
        "generation_times": []
    }

    pbar = tqdm(total=len(chunks), desc="Generating audio")

    models = []
    snac_models = []
    for i in range(available_gpus):
        with torch.device(f"cuda:{i}"):
            model, snac_model, tokenizer = getModels(MODEL_PATH)
            models.append(model)
            snac_models.append(snac_model)

    # Voice warmup
    warm_up_text = "Hi there, this is a warm up sentence so that the voice stabilizes from the beginning.  "
    for model in models:
        with torch.inference_mode():
            print(f"Warming up model in {model.device}")
            _ = model.generate(
                **tokenizer(build_prompt(tokenizer, warm_up_text, description), return_tensors="pt").to(model.device),
                max_new_tokens=1024,
                min_new_tokens=28,
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
                attn_implementation="flash_attention_2"
            )

    audio_path = outputPath + f"audios/{title}/"

    for gpu_id in range(available_gpus):
        t = threading.Thread(target=gpu_worker,
                             args=(f"cuda:{gpu_id}", q, models[gpu_id], snac_models[gpu_id], tokenizer, description,
                                   sharedData, lock, writer, audio_path, pbar))
        t.start()
        threads.append(t)

    q.join()

    for t in threads:
        t.join()

    writer.flush()
    writer.close()

    ordered_audios = sharedData["results"]
    ordered_indices = sorted(ordered_audios.keys())
    full_audio = [ordered_audios[idx] for idx in ordered_indices]
    saveAudio(audio_path, full_audio, title)


def gpu_worker(gpu_id, q, model, snac_model, tokenizer, description, sharedData, lock, writer, outputPath, pbar):
    device = torch.device(gpu_id)
    while True:
        try:
            idx, text = q.get(timeout=2)
        except queue.Empty:
            print(f"[{gpu_id}] No more prompts. Shutting down.")
            break

        if not text.strip():
            audio = np.zeros(int(0.2 * 24000))
            generation_time = 0
            audio_duration = 0
            # print(f"[{gpu_id}] Voice generation for part {idx} (para break)")
        else:
            # print(f"[{gpu_id}] Voice generation for part {idx} ...")
            start_time = time.time()
            audio = processVoice(model, device, tokenizer, snac_model, text, description, idx)
            generation_time = time.time() - start_time
            audio_duration = (len(audio) / 24000)
            # print(f"[{gpu_id}] Voice generation for part {idx} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")

        with lock:
            sharedData["results"][idx] = audio
            if text.strip():
                pbar.update(1)
                text_len = len(text)
                sharedData["global_step"] += 1
                step = sharedData["global_step"]
                sharedData["input_lengths"].append(text_len)
                sharedData["generation_times"].append(generation_time)
                if step > 1 and step % LOG_STEPS == 0:
                    writer.add_scalar("Evaluation/InputSize", text_len, step)
                    writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
                    writer.add_scalar("Performance/GenerationTime", generation_time, step)

                    rtf = (generation_time / audio_duration) if audio_duration > 0 else float('inf')
                    writer.add_scalar("Performance/RTF", rtf, step)

                    correlation = np.corrcoef(sharedData["input_lengths"], sharedData["generation_times"])[0, 1]
                    writer.add_scalar("Performance/InputDurationCorr", correlation, step)

                    if step % (LOG_STEPS * 2) == 0:
                        # Save partial audios
                        partial_audios = sharedData["results"]
                        partial_indices = sorted(partial_audios.keys())
                        partial_audio = [partial_audios[idx] for idx in partial_indices]

                        delete_previous_outputs(outputPath, step)
                        saveAudio(outputPath, partial_audio, f"partial_{step}")

        q.task_done()
        torch.cuda.empty_cache()