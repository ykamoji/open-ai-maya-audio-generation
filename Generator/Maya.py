import torch
import os
import glob
import time
import warnings
import soundfile as sf
import numpy as np
import threading
import queue
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from Generator.utils import createChunks
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

LOG_STEPS = 15


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
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "maya-research/maya1",
        cache_dir=MODEL_PATH,
        dtype="float16",
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "maya-research/maya1",
        cache_dir=MODEL_PATH,
        trust_remote_code=True
    )
    print(f"Model loaded: {len(tokenizer)} tokens in vocabulary")
    print("Loading SNAC audio decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    print("SNAC decoder loaded")
    return model, snac_model, tokenizer


def delete_previous_outputs(outputPath, step):
    files = glob.glob(outputPath + "partial_*.wav")
    files.sort(key=os.path.getmtime)
    files_to_delete = files[:-2]
    if files_to_delete and step > 0 and step % LOG_STEPS == 0:
        for f in files_to_delete:
            try:
                os.remove(f)
            except Exception as e:
                pass


def convert(Args, content, title):
    MayaArgs = Args.Generator.Maya

    MODEL_PATH = MayaArgs.ModelPath.__dict__[Args.Platform]

    model, snac_model, tokenizer = getModels(MODEL_PATH)

    description = getDescription(MayaArgs, title)

    outputPath = Args.Generator.AudioOutputPath.__dict__[Args.Platform]

    GPUCount = Args.Generator.GPU.__dict__[Args.Platform]

    maxTokens = Args.Generator.Maya.MaxTokens

    chunks = createChunks(content)

    if GPUCount > 0:
        print("Running in single GPU env.")
        multiGPU(chunks, description, maxTokens, model, outputPath, snac_model, title, tokenizer)
    else:
        print("Running in multi GPU env.")
        singleGPU(chunks, description, maxTokens, model, outputPath, snac_model, title, tokenizer)


def processVoice(model, device, tokenizer, snac_model, text, description, part, maxTokens):
    prompt = build_prompt(tokenizer, description, text)

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=maxTokens,  # Increase to let model finish naturally
            min_new_tokens=28,  # At least 4 SNAC frames
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,  # Prevent loops
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()

    if CODE_END_TOKEN_ID in generated_ids:
        eos_position = generated_ids.index(CODE_END_TOKEN_ID)
        print(f"Part {part} EOS token found at position {eos_position}/{len(generated_ids)}")
    else:
        print(f"Part {part} EOS token not found!")

    # Extract SNAC audio tokens
    snac_tokens = extract_snac_codes(generated_ids)

    if len(snac_tokens) < 7:
        print(f"Part {part} Error: Not enough SNAC tokens generated")
        # return

    levels = unpack_snac_from_7(snac_tokens)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
        for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()

    return audio


def singleGPU(chunks, description, maxTokens, model, outputPath, snac_model, title, tokenizer):
    audio_chunks = []
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    if torch.cuda.is_available():
        model.to("cuda")
        snac_model.to("cuda")
    for part, chunk in enumerate(chunks):
        # print(chunk)
        input_length = len(chunk)
        if input_length == 0:
            # Adding a pause between paras to keep the conversation seperate
            audio_chunks.append(np.zeros(int(0.125 * 24000)))
            print(f"Voice generation for part {step} (para break)")
            continue
        print(f"Voice generation for part {step} ...")
        start_time = time.time()
        audio = processVoice(model, "cuda", tokenizer, snac_model, chunk, description, part, maxTokens)
        generation_time = time.time() - start_time
        audio_duration = (len(audio) / 24000)
        print(f"Voice generation for part {step} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
        audio_chunks.append(audio)
        # Adding a pause between lines to keep the conversation consistent
        audio_chunks.append(np.zeros(int(0.1 * 24000)))

        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
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

            delete_previous_outputs(outputPath, step)
            partial_audio = np.concatenate(audio_chunks)
            file = outputPath + f"partial_{step}.wav"
            sf.write(file, partial_audio, 24000)
            print(f"Saving partial audio until {step}")

        step += 1
    writer.close()
    full_audio = np.concatenate(audio_chunks)
    file = outputPath + f"{title}.wav"
    sf.write(file, full_audio, 24000)
    print(f"Saved to {file}")


def multiGPU(chunks, description, maxTokens, model, outputPath, snac_model, title, tokenizer):
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")

    snac_model.to("cuda")
    available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    q = queue.Queue()

    for idx, text in enumerate(chunks):
        q.put((idx, text, description))

    lock = threading.Lock()
    threads = []

    sharedData = {
        "results": {},
        "global_step": 0,
        "logger": writer,
        "input_lengths": [],
        "generation_times": []
    }
    for gpu_id in available_gpus:
        t = threading.Thread(target=gpu_worker,
                             args=(gpu_id, q, model, snac_model, tokenizer, maxTokens, sharedData, lock, outputPath))
        t.start()
        threads.append(t)

    q.join()

    for t in threads:
        t.join()

    ordered_audios = sharedData["results"]
    ordered_indices = sorted(ordered_audios.keys())
    full_audio = np.concatenate([ordered_audios[idx] for idx in ordered_indices])
    file = outputPath + f"{title}.wav"
    sf.write(file, full_audio, 24000)
    print(f"Saved to {file}")


def gpu_worker(gpu_id, q, model, snac_model, tokenizer, maxTokens, sharedData, lock, outputPath):
    device = torch.device(gpu_id)
    model.to(device)

    while True:
        try:
            idx, text, description = q.get(timeout=3)
        except queue.Empty:
            print(f"[{gpu_id}] No more prompts. Shutting down.")
            break

        text_len = len(text)
        if text_len == 0:
            audio = np.zeros(int(0.125 * 24000))
            generation_time = 0
            audio_duration = 0
            print(f"Voice generation for part {idx} (para break)")
        else:
            print(f"Voice generation for part {idx} ...")
            start_time = time.time()
            audio = processVoice(model, device, tokenizer, snac_model, text, description, idx, maxTokens)
            generation_time = time.time() - start_time
            audio_duration = (len(audio) / 24000)
            print(f"Voice generation for part {idx} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
            audio = np.concatenate([audio, np.zeros(int(0.1 * 24000))])

        with lock:
            sharedData["results"][idx] = audio
            if text_len > 0:
                step = sharedData["global_step"]
                if sharedData["global_step"] > 0 and sharedData["global_step"] % LOG_STEPS == 0:
                    writer = sharedData["logger"]
                    writer.add_scalar("Evaluation/InputSize", text_len, step)
                    writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
                    writer.add_scalar("Performance/GenerationTime", generation_time, step)

                    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

                    sharedData["input_lengths"].append(text_len)
                    sharedData["generation_times"].append(generation_time)

                    writer.add_scalar("Performance/RTF", rtf, step)
                    correlation = np.corrcoef(sharedData["input_lengths"], sharedData["generation_times"])[0, 1]
                    writer.add_scalar("Performance/InputDurationCorr", correlation, step)

                    partial_audios = sharedData["results"]
                    partial_audios = sorted(partial_audios.keys())
                    partial_audio = np.concatenate([partial_audios[idx] for idx in partial_audios])

                    delete_previous_outputs(outputPath, step)
                    file = outputPath + f"partial_{step}.wav"
                    sf.write(file, partial_audio, 24000)
                    print(f"Saving partial audio until {step}")

                sharedData["global_step"] += 1
        q.task_done()
