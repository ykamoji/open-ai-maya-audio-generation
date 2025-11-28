import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import glob
import time
import warnings
import numpy as np
import multiprocessing as mp
from collections import deque
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
    print(f"\nDescription: {description}")
    return description


def convert(Args, pages, outputPath):
    platform = Args.Platform
    MayaArgs = Args.Generator.Maya
    MODEL_NAME = MayaArgs.ModelName.__dict__[platform]
    CACHE_PATH = MayaArgs.CachePath.__dict__[platform]

    tokenizer = getTokenizer(MODEL_NAME, CACHE_PATH, platform)

    GPUCount = torch.cuda.device_count()

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if GPUCount > 1:
        print("\nRunning in multi GPU env.")

        process_function = multiGPU
        args = {
            "GPUCount": GPUCount,
            "MODEL_NAME": MODEL_NAME,
            "CACHE_PATH": CACHE_PATH,
            "platform": platform,
            "outputPath": outputPath,
            "snac_model": getSnacModel(CACHE_PATH),
        }
    else:
        print("Running in CPU or single GPU env.")
        process_function = singleProcess
        args = {
            "model": getARModel(MODEL_NAME, CACHE_PATH, platform),
            "tokenizer": tokenizer,
            "snac_model": getSnacModel(CACHE_PATH),
            "outputPath": outputPath,
        }

    processed = 0
    for page in tqdm(pages, desc="Pages", ncols=120, position=0):
        try:
            chunks, tagged_list, para_breaks = batch_sentences(page['content'])

            # br = 0
            # for l, chunk in enumerate(chunks):
            #     if br < len(para_breaks) and l == para_breaks[br]:
            #         print("------Para break-------")
            #         br += 1
            #     print(chunk + f" Tagged: {tagged_list[l]}" )
            #
            # exit(1)

            description = getDescription(MayaArgs, page['title'])
            prompt_inputs = [
                (tokenizer(build_prompt(tokenizer, description, chunks[i]), return_tensors="pt"), tagged_list[i])
                for i in range(len(chunks))
            ]

            process_function(**args,
                             para_breaks=para_breaks,
                             tagged_list=tagged_list,
                             prompt_inputs=prompt_inputs,
                             title=page['title'])
            processed += 1
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}")

        clear_cache()

    return processed


def estimate_tokens(tok_len):
    if tok_len <= 60:
        return tok_len * 11 + 120
    elif tok_len < 75:
        return tok_len * 15 + 120
    else:
        return tok_len * 18 + 120


def processVoice(model, tokenizer, inputs, is_tagged, part):
    try:
        max_new_tokens = estimate_tokens(len(inputs['input_ids'][0]))
        start_time = time.time()
        while True:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs.to(model.device),
                    max_new_tokens=2048,
                    min_new_tokens=28,  # At least 4 SNAC frames
                    temperature=0.35 if is_tagged else 0.25,
                    top_p=0.92 if is_tagged else 0.8,
                    repetition_penalty=1.05 if is_tagged else 1.0,
                    do_sample=True,
                    eos_token_id=CODE_END_TOKEN_ID,  # Stop at end of speech token
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            generated_tokens = outputs[0, inputs['input_ids'].shape[1]:].tolist()
            if CODE_END_TOKEN_ID not in generated_tokens:
                max_new_tokens += 200
                print(f"\nPart {part}. EOS token not found. Trying again with {max_new_tokens}")
                del generated_tokens
            else:
                eos_position = generated_tokens.index(CODE_END_TOKEN_ID)
                print(f"Part {part} EOS token found at position {eos_position}/{len(generated_tokens)} for {len(inputs['input_ids'][0])}")
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


def singleProcess(model, snac_model, tokenizer, outputPath, para_breaks, tagged_list, prompt_inputs, title):
    input_lengths = []
    generation_times = []
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    step = 0
    model = model.to(getDevice())
    snac_model = snac_model.to(getDevice())
    audio_path = outputPath + f"audios/{title}/"
    for part, (inputs, is_tagged) in enumerate(tqdm(prompt_inputs, desc=f"{title}", ncols=90, position=1)):
        # print(chunk)
        # print(f"Voice generation for part {step}/{total} ...")
        generated_ids, (generation_time, audio_duration, rtf, tps) = processVoice(model, tokenizer, inputs, is_tagged, part)
        np.save(generated_ids, audio_path + f"part_{step}")
        del generated_ids
        # print(f"Voice generation for part {step} ({audio_duration:.2f} sec) in {generation_time:.2f} sec")
        input_length = len(inputs['input_ids'][0])
        input_lengths.append(input_length)
        generation_times.append(generation_time)
        step += 1
        if step > 1 and step % LOG_STEPS == 0:
            writer.add_scalar("Evaluation/InputSize", input_length, step)
            writer.add_scalar("Evaluation/AudioDuration", audio_duration, step)
            writer.add_scalar("Performance/GenerationTime", generation_time, step)
            writer.add_scalar("Performance/RTF", rtf, step)
            writer.add_scalar("Performance/TPS", tps, step)
            if step > 2:
                correlation = np.corrcoef(input_lengths, generation_times)[0, 1]
                writer.add_scalar("Performance/InputDurationCorr", correlation, step)

            # if step % (LOG_STEPS * 2) == 0:
                # delete_previous_outputs(audio_path, step)
                # create_audio(generated_outputs, snac_model, audio_path, f"partial_{step}")

    writer.close()

    part_files = _gather_sorted_part_files(audio_path, title)

    generated_tokens_full = [np.load(file) for file in part_files]

    create_audio(generated_tokens_full, snac_model, audio_path, para_breaks, tagged_list, title)


def multiGPU(GPUCount, MODEL_NAME, CACHE_PATH, platform, snac_model, outputPath, para_breaks, tagged_list, prompt_inputs, title):

    mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    task_q = manager.Queue()
    metrics_q = manager.Queue()
    done_event = manager.Event()

    audio_path = outputPath + f"audios/{title}/"

    # push tasks
    for idx, inp in enumerate(prompt_inputs):
        task_q.put((idx, inp))

    # sentinels
    for _ in range(GPUCount):
        task_q.put(None)

    # start aggregator process
    agg_proc = mp.Process(
        target=metric_worker,
        args=(metrics_q, outputPath, title, done_event, len(prompt_inputs)),
        daemon=True,
    )
    agg_proc.start()

    # start worker processes (one per GPU)
    procs = []
    for gpu_id in range(GPUCount):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, MODEL_NAME, CACHE_PATH, platform, task_q, metrics_q, audio_path),
        )
        p.start()
        procs.append(p)

    # wait for workers to finish
    for p in procs:
        p.join()

    # tell aggregator to finish and wait
    done_event.set()
    agg_proc.join()

    # ---------- All AR parts are saved on disk now ----------
    # Gather and sort parts
    part_files = _gather_sorted_part_files(outputPath+"/audios/", title)

    # Concatenate parts into one generated_ids object
    generated_tokens_full = [np.load(file) for file in part_files]

    device = torch.device("cuda:0")
    snac_model.to(device)
    snac_model.eval()
    for p in snac_model.parameters():
        p.requires_grad = False

    completed = create_audio(generated_tokens_full, snac_model, audio_path, para_breaks, tagged_list, title)

    # cleanup
    try:
        del snac_model
        del generated_tokens_full
        torch.cuda.empty_cache()
    except Exception:
        pass

    if completed:
        for file in part_files:
            try:
                os.remove(file)
            except Exception:
                pass

    clear_cache()


def gpu_worker(gpu_id, MODEL_NAME, CACHE_PATH, platform, task_q, metrics_q, outputPath):

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
    print(f"Voice warm up completed for GPU {gpu_id}")

    while True:
        item = task_q.get()
        if item is None:
            break
        idx, (inputs, is_tagged) = item
        try:

            generated_tokens, meta = processVoice(model, tokenizer, inputs, is_tagged, idx)

            np.save(outputPath + f"part_{idx}.npy", generated_tokens)

            try:
                del generated_tokens
                torch.cuda.empty_cache()
            except Exception:
                pass

            metric = {
                "idx": idx,
                "text_len": len(inputs["input_ids"][0]),
                "generation_time": meta[0],
                "audio_duration": meta[1],
                "rtf": meta[2],
                "tps": meta[3],
            }
            metrics_q.put(metric)

        except Exception as e:
            metrics_q.put({"idx": idx, "error": str(e)})
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _gather_sorted_part_files(parts_dir, title):

    files = [file for file in glob.glob(parts_dir + f"{title}/*.npy") if "part_" in file]

    def idx_from_name(p):
        base = os.path.basename(p)
        name = os.path.splitext(base)[0]
        try:
            return int(name.split("_")[1])
        except Exception:
            return 10**9

    return sorted(files, key=idx_from_name)


def metric_worker(metrics_q, outputPath, title, done_event, total_parts, log_steps=20):

    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    input_lengths = []
    generation_times = []
    received = 0
    errors = 0
    recent = deque(maxlen=1000)
    pbar = tqdm(total=total_parts, desc=f"{title}", ncols=90, position=1)
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

            if step > 0 and step % log_steps == 0:
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