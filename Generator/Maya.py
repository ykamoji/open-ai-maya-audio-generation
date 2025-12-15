import os
import torch
import glob
import time
import warnings
import numpy as np
import sys
import multiprocessing as mp
from collections import deque
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from Emotions.utils import getDevice, clear_cache
from Generator.utils import getARModel, getTokenizer, load_dialogues, move_edited_dialogues

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

LOG_STEPS = 5


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
    snac_count = sum(1 for t in generated_ids if SNAC_MIN_ID <= t <= SNAC_MAX_ID)
    duration_sec = snac_count * 0.01194 + 0.07499
    return duration_sec


def getDescription(MayaArgs, title):
    description = ""
    for character in MayaArgs.Characters:
        if character.Name in title:
            description = character.Description
    print(f"\nDescription: {description}")
    return description


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


def convert(Args, pages, outputPath):
    platform = Args.Platform
    MayaArgs = Args.Generator.Maya
    MODEL_NAME = MayaArgs.ModelName.__dict__[platform]
    CACHE_PATH = MayaArgs.CachePath.__dict__[platform]

    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH, platform)

    GPUCount = torch.cuda.device_count()

    if GPUCount > 1:
        print("\nRunning in multi GPU env.")
        mp.set_start_method("spawn", force=True)
        process_function = multiGPU
        args = {
            "GPUCount": GPUCount,
            "MODEL_NAME": MODEL_NAME,
            "CACHE_PATH": CACHE_PATH,
            "platform": platform,
            "outputPath": outputPath,
        }
    else:
        print("Running in CPU or single GPU env.")
        process_function = singleProcess
        args = {
            "model": getARModel(MODEL_NAME, CACHE_PATH, platform),
            "tokenizer": tokenizer,
            "outputPath": outputPath,
        }

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=120, position=0, file=sys.stdout):
        try:
            # chunks, tagged_list, para_breaks, broken_paras = batch_sentences(page['content'])
            val = load_dialogues(Args.Graph.NotebookName, Args.Graph.SectionName, page, Args.forceUpdate, Args.correctVoice)
            chunks, tagged_list, para_breaks, broken_paras, edit_present = val

            if broken_paras:
                continue

            chunks = chunks[:5]

            ## when partial parts were interrupted from last session
            start = 0
            part_files = _gather_sorted_part_files(outputPath + "/audios/", page['title'])
            if part_files:
                chunks = chunks[len(part_files) - 1:]
                start = len(part_files) - 1

            # continue

            description = getDescription(MayaArgs, page['title'])
            inputs = [
                (
                    tokenizer(build_prompt(tokenizer, description, chunks[i]["dialogue"]), return_tensors="pt"),
                    tagged_list[i],
                    chunks[i]["part"],
                    chunks[i]["updated"],
                )
                for i in range(len(chunks))
            ]

            process_function(**args,
                             para_breaks=para_breaks,
                             tagged_list=tagged_list,
                             inputs=inputs,
                             start=start,
                             edit_present=edit_present,
                             title=page['title'])
            processed += 1

            if edit_present:
                move_edited_dialogues(Args.Graph, page)

        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}")

        clear_cache()

    return processed


def max_new_token_boundaries(input_length):
    if input_length < 60:
        return 2048
    elif input_length < 100:
        return 2600
    else:
        return 4600


def processVoice(model, tokenizer, prompt_input, is_tagged, part):
    try:
        max_new_tokens = max_new_token_boundaries(len(prompt_input["input_ids"][0]))
        start_time = time.time()
        while True:
            with torch.inference_mode():
                outputs = model.generate(
                    **prompt_input.to(model.device),
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=28,  # At least 4 SNAC frames
                    temperature=0.4 if is_tagged else 0.25,
                    top_p=0.9 if is_tagged else 0.98,
                    repetition_penalty=1.10,
                    do_sample=True,
                    eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            generated_tokens = outputs[0, prompt_input['input_ids'].shape[1]:].tolist()
            if CODE_END_TOKEN_ID not in generated_tokens:
                max_new_tokens += 500
                print(f"\nPart {part} ({len(prompt_input['input_ids'][0])}). EOS token not found. Trying again with {max_new_tokens}\n")
                del outputs
                del generated_tokens
            else:
                # eos_position = generated_tokens.index(CODE_END_TOKEN_ID)
                # print(f"Part {part} EOS token found at position {eos_position}/{len(generated_tokens)} for {len(inputs['input_ids'][0])}")
                break

        generation_time = time.time() - start_time
        audio_duration = estimate_audio_duration(generated_tokens)
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
        tps = len(generated_tokens) / audio_duration if audio_duration > 0 else float("inf")
    except Exception as e:
        print(f"Model error: {e}")
        generated_tokens = []
        generation_time = 0
        audio_duration = 0
        rtf = float('inf')
        tps = float("nan")

    return generated_tokens, (generation_time, audio_duration, rtf, tps)


def singleProcess(model, tokenizer, outputPath, para_breaks, tagged_list, inputs, start, edit_present, title):
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    model = model.to(getDevice())
    audio_path = outputPath + f"audios/{title}/"
    Path(audio_path).mkdir(parents=True, exist_ok=True)
    for prompt_input, is_tagged, part_id, updated in tqdm(inputs, desc=f"{title}", ncols=90, position=1, file=sys.stdout, initial=start):

        # print(f"Voice generation for part {step}/{total} ...")

        # For post editing. Skipping parts which don't need to be generated.
        if edit_present and not updated:
            continue

        generated_ids, (generation_time, audio_duration, rtf, tps) = processVoice(model, tokenizer, prompt_input, is_tagged, part_id)
        if edit_present:
            np.save(audio_path + f"edited_{part_id}.npy", generated_ids)
        else:
            np.save(audio_path + f"part_{part_id}.npy", generated_ids)
        del generated_ids

        # print(f"Voice generation for part {step} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
        input_length = len(prompt_input['input_ids'][0])
        input_lengths.append(input_length)
        generation_times.append(generation_time)
        step += 1
        if step % LOG_STEPS == 0:
            writer.add_scalar("Evaluation/InputSize", input_length, step)
            writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
            writer.add_scalar("Performance/GenerationTime", generation_time, step)
            writer.add_scalar("Performance/RTF", rtf, step)
            writer.add_scalar("Performance/TPS", tps, step)
            if step > 2:
                correlation = np.corrcoef(input_lengths, generation_times)[0, 1]
                writer.add_scalar("Performance/InputDurationCorr", correlation, step)

    writer.close()

    if not edit_present:
        combine(audio_path, outputPath, para_breaks, tagged_list, title)

    torch.cuda.empty_cache()


def multiGPU(GPUCount, MODEL_NAME, CACHE_PATH, platform, outputPath, para_breaks, tagged_list,
             inputs, start, edit_present, title):

    task_q = mp.Queue()
    metrics_q = mp.Queue()
    done_event = mp.Event()

    audio_path = outputPath + f"audios/{title}/"

    # push tasks
    for input in inputs:
        task_q.put(input)

    # sentinels
    for _ in range(GPUCount):
        task_q.put(None)

    # start aggregator process
    agg_proc = mp.Process(
        target=metric_worker,
        args=(metrics_q, outputPath, title, done_event, start, len(inputs)),
        daemon=True,
    )
    agg_proc.start()

    # start worker processes (one per GPU)
    procs = []
    for gpu_id in range(GPUCount):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, MODEL_NAME, CACHE_PATH, platform, task_q, metrics_q, audio_path, edit_present),
        )
        p.start()
        procs.append(p)

    # wait for workers to finish
    for p in procs:
        p.join()

    # tell aggregator to finish and wait
    done_event.set()
    agg_proc.join()

    if not edit_present:
        combine(audio_path, outputPath, para_breaks, tagged_list, title)

    torch.cuda.empty_cache()


def combine(audio_path, outputPath, para_breaks, tagged_list, title):
    # ---------- All AR parts are saved on disk now ----------
    # Gather and sort parts
    part_files = _gather_sorted_part_files(outputPath + "/audios/", title)
    # Concatenate parts into one generated_ids object
    generated_tokens_full = [np.load(file) for file in part_files]
    completed = save_snac_tokens(generated_tokens_full, audio_path, para_breaks, tagged_list, title)
    # cleanup
    try:
        del generated_tokens_full
    except Exception:
        pass
    if completed:
        for file in part_files:
            try:
                os.remove(file)
            except Exception:
                pass


def gpu_worker(gpu_id, MODEL_NAME, CACHE_PATH, platform, task_q, metrics_q, outputPath, edit_present):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    model = getARModel(MODEL_NAME, CACHE_PATH, platform, f"cuda:{gpu_id}")
    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH, platform)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Voice warmup
    try:
        dummy = torch.randint(0, 30000, (1, 5), dtype=torch.long, device=f"cuda:{gpu_id}")
        with torch.inference_mode():
            _ = model.model.layers[0](model.model.embed_tokens(dummy))
    except Exception:
        pass
    print(f"\nVoice warm up completed for GPU {gpu_id}\n")

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    while True:
        item = task_q.get()
        if item is None:
            break
        prompt_input, is_tagged, part_id, updated = item

        if edit_present and not updated:
            continue

        try:
            generated_tokens, meta = processVoice(model, tokenizer, prompt_input, is_tagged, part_id)
            Path(outputPath).mkdir(parents=True, exist_ok=True)
            if edit_present:
                np.save(outputPath + f"edited_{part_id}.npy", generated_tokens)
            else:
                np.save(outputPath + f"part_{part_id}.npy", generated_tokens)

            try:
                del generated_tokens
            except Exception:
                pass

            metric = {
                "idx": part_id,
                "text_len": len(prompt_input["input_ids"][0]),
                "generation_time": meta[0],
                "audio_duration": meta[1],
                "rtf": meta[2],
                "tps": meta[3],
            }
            metrics_q.put(metric)

        except Exception as e:
            metrics_q.put({"idx": part_id, "error": str(e)})


def idx_from_name(p):
    base = os.path.basename(p)
    name = os.path.splitext(base)[0]
    try:
        return int(name.split("_")[1])
    except Exception:
        return 10 ** 9


def _gather_sorted_part_files(parts_dir, title):
    files = [file for file in glob.glob(parts_dir + f"{title}/*.npy") if "part_" in file]

    if len(files) == 0:
        return []

    return sorted(files, key=idx_from_name)


def metric_worker(metrics_q, outputPath, title, done_event, start, total_parts, log_steps=5):
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    input_lengths = []
    generation_times = []
    received = 0
    errors = 0
    recent = deque(maxlen=1000)
    pbar = tqdm(total=total_parts, desc=f"{title}", ncols=90, position=1, file=sys.stdout, initial=start)
    while True:
        try:
            metric = metrics_q.get(timeout=1.0)
            if "error" in metric:
                errors += 1
                writer.add_text("Errors", f"idx {metric.get('idx')}: {metric['error']}", received)
                continue
            received += 1
            pbar.update(1)
            input_lengths.append(metric.get("text_len", 0))
            generation_times.append(metric.get("generation_time", 0.0))
            recent.append(metric)
            step = received

            if step % log_steps == 0:
                writer.add_scalar("Evaluation/InputSize", metric.get("text_len", 0), step)
                writer.add_scalar("Evaluation/AudioDuration", metric.get("audio_duration", 0), step)
                writer.add_scalar("Performance/GenerationTime", metric.get("generation_time", 0), step)
                try:
                    corr = float(np.corrcoef(input_lengths, generation_times)[0, 1]) if len(
                        input_lengths) >= 2 else 0.0
                except Exception:
                    corr = 0.0
                writer.add_scalar("Performance/InputDurationCorr", corr, step)

        except Exception:
            if done_event.is_set():
                while not metrics_q.empty():
                    metric = metrics_q.get_nowait()
                    if "error" in metric:
                        errors += 1
                    else:
                        received += 1
                break
            #

    writer.add_text("Summary", f"Received {received} parts with {errors} errors", received)
    writer.flush()
    writer.close()
    pbar.close()
