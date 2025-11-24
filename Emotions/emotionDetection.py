import torch
import inspect
from tqdm import tqdm
from Emotions.utils import *
from utils import updateCache


def get_context(i, sentences):
    prev_s = sentences[i - 1] if i - 1 >= 0 else ""
    curr_s = sentences[i]
    next_s = sentences[i + 1] if i + 1 < len(sentences) else ""
    return prev_s, curr_s, next_s


DETECTION_PREFIX = inspect.cleandoc(f"""
        You are a expert TTS emotion cue detector.

        Use the PREVIOUS and NEXT sentences ONLY as context to understand the emotional tone of the TARGET sentence.

        Allowed TAGS: {TTS_TAGS_STR}

        IMPORTANT RULES:
        1. Many TARGET sentences are NEUTRAL and must return an empty tag list.
        2. If the TARGET sentence expresses no emotion, no vocal tension, and no implied emotional state, you MUST return {{"tags": []}}.
        3. Do NOT guess. If unsure, return {{"tags": []}}.
        4. Only detect emotion if it is clearly present INSIDE the TARGET sentence. 
           - Emotion in context (previous/next) does NOT count unless the TARGET sentence continues it.
        5. NEVER output any tag not in Allowed TAGS.

        TASK:
        1. Identify all tags that match the emotional tone of the TARGET sentence.
        2. Rank them by strength.
        3. Return ONLY the top two tags (0–2 tags).

        OUTPUT FORMAT (STRICT):
            - If 0 tags match → {{"tags": []}}
            - If 1 tag matches → {{"tags": ["TAG1"]}}
            - If 2+ tags match → {{"tags": ["TAG1", "TAG2"]}}

        EXAMPLE 1 (emotional)
        PREVIOUS: He clenched the letter tightly.
        TARGET: “Just leave me alone,” he snapped.
        NEXT: She stepped away from him.
        OUTPUT: {{"tags": ["ANGRY"]}}

        EXAMPLE 2 (neutral)
        PREVIOUS: She opened the old wooden door.
        TARGET: The hallway stretched out before her.
        NEXT: A faint draft brushed past.
        OUTPUT: {{"tags": []}}

        EXAMPLE 3 (hard neutral)
        PREVIOUS: He looked at her for a long moment.
        TARGET: She finally nodded and walked away.
        NEXT: He watched her disappear around the corner.
        OUTPUT: {{"tags": []}}

        EXAMPLE 4 (context emotional but target neutral)
        PREVIOUS: His voice trembled as he spoke.
        TARGET: The candle flickered on the table.
        NEXT: She exhaled slowly.
        OUTPUT: {{"tags": []}}

        EXAMPLE 5 (subtle emotional cue)
        PREVIOUS: The room fell quiet.
        TARGET: She hesitated before reaching for the door.
        NEXT: No one said a word.
        OUTPUT: {{"tags": ["EXHALE"]}}

        PREVIOUS SENTENCE:
        """) + "\n"


def build_dynamic_suffix(prev_s, curr_s, next_s):
    return inspect.cleandoc(f"""
        {prev_s}

        TARGET SENTENCE:
        {curr_s}

        NEXT SENTENCE:
        {next_s}

        Answer:
    """)


# DETECTION_STATIC_IDS = None
DETECTION_STATIC_MASK = None
DETECTION_STATIC_BATCH_PAST = None


def init_detection_prefix_cache(model, tokenizer, BATCH_SIZE):
    global DETECTION_STATIC_MASK, DETECTION_STATIC_BATCH_PAST

    if DETECTION_STATIC_MASK is not None and DETECTION_STATIC_BATCH_PAST is not None:
        return  # already initialized

    device = next(model.parameters()).device

    enc = tokenizer(DETECTION_PREFIX, return_tensors="pt").to(device)
    static_ids = enc["input_ids"]  # (1, prefix_len)
    DETECTION_STATIC_MASK = enc["attention_mask"]  # (1, prefix_len)

    with torch.inference_mode():
        out = model(
            input_ids=static_ids,
            attention_mask=DETECTION_STATIC_MASK,
            use_cache=True,
        )

    # detach so we don't track autograd history
    detection_static_past = tuple((k.detach(), v.detach()) for k, v in out.past_key_values)

    DETECTION_STATIC_BATCH_PAST = repeat_past_kv(detection_static_past, BATCH_SIZE)


def detect_batch(title, indices, sentences, model, tokenizer, BATCH_SIZE: int = 20):
    global DETECTION_STATIC_MASK, DETECTION_STATIC_BATCH_PAST

    device = next(model.parameters()).device
    init_detection_prefix_cache(model, tokenizer, BATCH_SIZE)

    outputs = ["" for _ in range(len(indices))]

    for start in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"{title}", ncols=90, position=1):
        chunk = indices[start:start + BATCH_SIZE]

        dynamic_prompts = []
        for idx in chunk:
            prev_s, curr_s, next_s = get_context(idx, sentences)
            dynamic_prompts.append(build_dynamic_suffix(prev_s, curr_s, next_s))

        dyn = tokenizer(dynamic_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        dyn_ids = dyn["input_ids"]  # (B, dyn_len)
        dyn_mask = dyn["attention_mask"]  # (B, dyn_len)

        batch_size = dyn_ids.size(0)
        # prefix_len = DETECTION_STATIC_IDS.size(1)

        # full attention mask = [prefix_mask, dyn_mask]
        static_mask = DETECTION_STATIC_MASK.repeat(batch_size, 1)  # (B, prefix_len)
        full_mask = torch.cat([static_mask, dyn_mask], dim=1)  # (B, prefix_len + dyn_len)

        # repeat prefix KV across batch

        past_batch_val = DETECTION_STATIC_BATCH_PAST
        if batch_size < BATCH_SIZE:
            past_batch_val = slice_prefix_kv(DETECTION_STATIC_BATCH_PAST, batch_size)

        # logger.debug("Running emotion detection for:")
        # for idx in chunk:
        #     logger.debug(f'  "{sentences[idx]}"')

        try:
            out_ids = fast_generate(
                model,
                dynamic_ids=dyn_ids,
                attention_mask=full_mask,
                max_new_tokens=50,
                eos_token_id=tokenizer.eos_token_id,
                past_key_values=past_batch_val,
            )
            batch_outputs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"Exception during inference: {e}. Model cannot detect emotions.")
            batch_outputs = [""] * len(chunk)
        finally:
            out_ids = None
            static_mask = None
            full_mask = None

        # Parse tags for each sentence
        for local_i, idx in enumerate(chunk):
            try:
                start_pos = batch_outputs[local_i].find("Answer:")
                if start_pos == -1:
                    parsed_output = ""
                else:
                    parsed_output = batch_outputs[local_i][start_pos:]
                outputs[indices.index(idx)] = parsed_output
            except Exception as e:
                print(f"Exception while parsing tags: {e}")
                outputs[indices.index(idx)] = ""

        clear_cache()

    return outputs


def detectEmotions(Args, pages, notebook_name, section_name, EMOTION_CACHE):
    model, tokenizer = getModelAndTokenizer(Args)

    progress = 0
    for page in tqdm(pages, desc="Pages", ncols=100, position=0):
        content = page['content']
        try:
            lines = []
            para_breaks = []
            for para_no, paragraph in enumerate(content):
                if para_no > 0:
                    para_breaks.append(len(lines))
                lines.extend(split_sentences(paragraph))

            indices = list(range(len(lines)))
            output = detect_batch(page["title"], indices, lines, model, tokenizer)
            if output:
                line_outputs = []
                for line_idx, o in enumerate(output):
                    line_outputs.append(
                        {"line": lines[line_idx],
                         "output": output[line_idx],
                         }
                    )
                for brk in reversed(para_breaks):
                    line_outputs.insert(brk, {"line":""})

                EMOTION_CACHE[notebook_name][section_name][page["title"]] = line_outputs
                updateCache('emotionCache.json', EMOTION_CACHE)
                progress += 1
            else:
                print(f"Something went wrong while detecting emotions for {page['title']}.")
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}.")

    DETECTION_STATIC_IDS = None
    DETECTION_STATIC_MASK = None
    DETECTION_STATIC_BATCH_PAST = None
    clear_cache()

    return progress
