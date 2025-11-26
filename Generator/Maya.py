import torch
import os
import glob
import time
import warnings
import numpy as np
import threading
import queue
import gc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Emotions.utils import getDevice, clear_cache
from Generator.decoder import create_audio
from Generator.utils import batch_sentences

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

LOG_STEPS = 4


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


def convert(model, snac_model, tokenizer, MayaArgs, content, title, outputPath):

    GPUCount = 0
    if torch.cuda.is_available():
        GPUCount = torch.cuda.device_count()

    chunks = batch_sentences(content)

    torch.manual_seed(0)

    description = getDescription(MayaArgs, title)

    prompt_inputs = [
        tokenizer(build_prompt(tokenizer, description, chunk), return_tensors="pt")
        for chunk in chunks
    ]

    if GPUCount > 1:
        print("Running in multi GPU env.")
        # multiGPU(chunks, description, outputPath, title, MODEL_PATH)
    else:
        print("Running in CPU or single GPU env.")
        singleProcess(prompt_inputs, model, outputPath, snac_model, title, tokenizer)


def processVoice(model, tokenizer, inputs, part):
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs.to(model.device),
            max_new_tokens=len(inputs['input_ids'][0]) * 18 * 1.4,
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

    # tps = len(generated_ids) / audio_duration if audio_duration > 0 else float("nan")

    del outputs
    clear_cache()

    return generated_ids, generation_time, audio_duration, rtf


def singleProcess(prompt_inputs, model, outputPath, snac_model, title, tokenizer):
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    model = model.to(getDevice())
    snac_model = snac_model.to(getDevice())
    audio_path = outputPath + f"audios/{title}/"
    generated_outputs = []
    for part, inputs in enumerate(tqdm(prompt_inputs, desc="Generating audio")):
        # print(chunk)
        # print(f"Voice generation for part {step}/{total} ...")
        generated_ids, generation_time, audio_duration, rtf = processVoice(model, tokenizer, inputs, part)
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

            if step > 2:
                correlation = np.corrcoef(input_lengths, generation_times)[0, 1]
                writer.add_scalar("Performance/InputDurationCorr", correlation, step)

            delete_previous_outputs(audio_path, step)
            create_audio(generated_outputs, snac_model, audio_path, f"partial_{step}")

        step += 1
    writer.close()

    create_audio(generated_outputs, snac_model, audio_path, title)



# def multiGPU(chunks, description, outputPath, title, MODEL_PATH):
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
#             model, snac_model, tokenizer = getModels(MODEL_PATH)
#             models.append(model)
#             snac_models.append(snac_model)
#
#     # Voice warmup
#     warm_up_text = "Hi there, this is a warm up sentence so that the voice stabilizes from the beginning.  "
#     for model in models:
#         with torch.inference_mode():
#             print(f"Warming up model in {model.device}")
#             _ = model.generate(
#                 **tokenizer(build_prompt(tokenizer, warm_up_text, description), return_tensors="pt").to(model.device),
#                 max_new_tokens=1024,
#                 min_new_tokens=28,
#                 temperature=0.4,
#                 top_p=0.9,
#                 repetition_penalty=1.1,
#                 do_sample=True,
#                 eos_token_id=CODE_END_TOKEN_ID,
#                 pad_token_id=tokenizer.pad_token_id,
#                 attn_implementation="flash_attention_2"
#             )
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