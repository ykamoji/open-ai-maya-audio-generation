import torch
import os
import glob
import time
import warnings
import numpy as np
import threading
import queue
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Emotions.utils import getDevice, clear_cache
from Generator.decoder import create_audio
from Generator.utils import batch_sentences, getARModel, getSnacModel, getTokenizer

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


def estimate_audio_duration(generated_ids):
    snac_count = len([SNAC_MIN_ID <= t <= SNAC_MAX_ID for t in generated_ids])
    duration_sec = snac_count * 0.01194 + 0.07499
    return duration_sec


def getDescription(MayaArgs, title):
    description = ""
    for character in MayaArgs.Characters:
        if character.Name in title:
            description = character.Description
    print(f"Description: {description}")
    return description


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


def convert(Args, pages, outputPath):

    platform = Args.Platform
    MayaArgs = Args.Generator.Maya
    MODEL_NAME = MayaArgs.ModelName.__dict__[platform]
    CACHE_PATH = MayaArgs.CachePath.__dict__[platform]

    snac_model = getSnacModel(CACHE_PATH)
    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH, platform)

    GPUCount = torch.cuda.device_count()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if GPUCount > 1:
        print("Running in multi GPU env.")
        models = []
        for i in range(GPUCount):
            model = getARModel(MODEL_NAME, CACHE_PATH, platform, f"cuda:{i}")
            # Voice warmup
            try:
                dummy = torch.randint(0, 30000, (1, 5), dtype=torch.long, device=f"cuda:{i}")
                with torch.inference_mode():
                    _ = model.model.layers[0](model.model.embed_tokens(dummy))
            except Exception:
                pass
            print(f"Voice warm up completed for GPU {i}")
            models.append(model)
    else:
        print("Running in CPU or single GPU env.")
        models = [getARModel(MODEL_NAME, CACHE_PATH, platform)]

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=120, position=0):

        try:

            chunks = batch_sentences(page['content'])
            description = getDescription(MayaArgs, page['title'])
            prompt_inputs = [
                tokenizer(build_prompt(tokenizer, description, chunk), return_tensors="pt")
                for chunk in chunks
            ]
            if GPUCount > 1:
                multiGPU(GPUCount, models, snac_model, tokenizer, prompt_inputs, outputPath, page['title'])
            else:
                singleProcess(prompt_inputs, models[0], snac_model, tokenizer, outputPath, page['title'])

            processed += 1
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}")

        clear_cache()


    return processed


def processVoice(model, tokenizer, inputs, part):
    try:
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs.to(model.device),
                max_new_tokens=int(len(inputs['input_ids'][0]) * 18 * 1.4),
                min_new_tokens=28,  # At least 4 SNAC frames
                temperature=0.4,
                top_p=0.9,
                repetition_penalty=1.1,  # Prevent loops
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        generation_time = time.time() - start_time
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
        if CODE_END_TOKEN_ID not in generated_ids:
            # eos_position = generated_ids.index(CODE_END_TOKEN_ID)
            # print(f"Part {part} EOS token found at position {eos_position}/{len(generated_ids)}")
            print(f"Part {part}. EOS token not found.")
        audio_duration = estimate_audio_duration(generated_ids)
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        tps = len(generated_ids) / audio_duration if audio_duration > 0 else float("nan")
    except Exception as e:
        print(f"AR model error: {e}")
        generated_ids = 0
        generation_time = 0
        audio_duration = 0
        rtf = float('inf')
        tps = float("nan")

    return generated_ids, generation_time, audio_duration, rtf, tps


def singleProcess(prompt_inputs, model, snac_model, tokenizer, outputPath, title):
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    model = model.to(getDevice())
    snac_model = snac_model.to(getDevice())
    audio_path = outputPath + f"audios/{title}/"
    generated_outputs = []
    for part, inputs in enumerate(tqdm(prompt_inputs, desc="Generating audio", ncols=90, position=1)):
        # print(chunk)
        # print(f"Voice generation for part {step}/{total} ...")
        generated_ids, generation_time, audio_duration, rtf, tps = processVoice(model, tokenizer, inputs, part)
        generated_outputs.append(generated_ids)
        # print(f"Voice generation for part {step} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
        input_length = len(inputs['input_ids'][0])
        input_lengths.append(input_length)
        generation_times.append(generation_time)
        if step % LOG_STEPS == 0:
            writer.add_scalar("Evaluation/InputSize", input_length, step)
            writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
            writer.add_scalar("Performance/GenerationTime", generation_time, step)
            writer.add_scalar("Performance/RTF", rtf, step)
            writer.add_scalar("Performance/TPS", tps, step)
            if step > 2:
                correlation = np.corrcoef(input_lengths, generation_times)[0, 1]
                writer.add_scalar("Performance/InputDurationCorr", correlation, step)

            if step % (LOG_STEPS * 2) == 0:
                delete_previous_outputs(audio_path, step)
                create_audio(generated_outputs, snac_model, audio_path, f"partial_{step}")

        step += 1
    writer.close()

    create_audio(generated_outputs, snac_model, audio_path, title)


def multiGPU(GPUCount, models, snac_model, tokenizer, prompt_inputs, outputPath, title):

    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")

    q = queue.Queue()

    count = 0
    for idx, inputs in enumerate(prompt_inputs):
        q.put((idx, inputs))
        count += 1

    # print(f"{title}: {count}")

    lock = threading.Lock()
    threads = []

    sharedData = {
        "results": {},
        "global_step": 0,
        "input_lengths": [],
        "generation_times": []
    }

    audio_path = outputPath + f"audios/{title}/"

    pbar = tqdm(total=count, desc=f"{title}", ncols=90, position=0)

    for gpu_id in range(GPUCount):
        t = threading.Thread(target=gpu_worker,
                             args=(f"cuda:{gpu_id}", q, models[gpu_id], snac_model, tokenizer, sharedData, lock, writer, audio_path, pbar))
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
    create_audio(full_audio, snac_model, audio_path, title)
    clear_cache()


def gpu_worker(gpu_id, q, model, snac_model, tokenizer, sharedData, lock, writer, outputPath, pbar):
    while True:
        try:
            idx, inputs = q.get(timeout=2)
        except queue.Empty:
            print(f"[{gpu_id}] No more prompts. Shutting down.")
            break

        # print(f"[{gpu_id}] Voice generation for part {idx} ...")
        start_time = time.time()
        generated_ids, generation_time, audio_duration, rtf, tps = processVoice(model, tokenizer, inputs, idx)
        generation_time = time.time() - start_time
        audio_duration = estimate_audio_duration(generated_ids)
        # print(f"[{gpu_id}] Voice generation for part {idx} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")

        with lock:
            sharedData["results"][idx] = generated_ids
            pbar.update(1)
            text_len = len(inputs["input_ids"][0])
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
                writer.add_scalar("Performance/TPS", tps, step)
                correlation = np.corrcoef(sharedData["input_lengths"], sharedData["generation_times"])[0, 1]
                writer.add_scalar("Performance/InputDurationCorr", correlation, step)

                if step % (LOG_STEPS * 2) == 0:
                    # Save partial audios
                    partial_audios = sharedData["results"]
                    partial_indices = sorted(partial_audios.keys())
                    partial_audio = [partial_audios[idx] for idx in partial_indices]
                    try:
                        delete_previous_outputs(outputPath, step)
                        create_audio(partial_audio, snac_model, outputPath, f"partial_{step}")
                    except Exception as e:
                        print(f"Snac decoding error: {e}")

        q.task_done()