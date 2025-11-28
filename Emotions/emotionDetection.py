import inspect
import json
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Emotions.utils import *
from utils import updateCache


def get_context(i, sentences):
    prev_s = sentences[i - 1] if i - 1 >= 0 else ""
    curr_s = sentences[i]
    next_s = sentences[i + 1] if i + 1 < len(sentences) else ""
    return prev_s, curr_s, next_s


def extract_emotion_tags(text):
    match = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', text)
    if match:
        inner = match.group(1)
        tags = str(re.findall(r'"([^"]+)"', inner)).replace("'", '"')
        # return [f"[{tag.upper()}]" for tag in json.loads(tags) if f"[{tag.upper()}]" in TTS_TAGS]
        valid_tags = []
        invalid_tags = []
        for tag in json.loads(tags):
            t = f"[{tag.upper()}]"
            if t in TTS_TAGS_SET:
                valid_tags.append(t)
            else:
                invalid_tags.append(t)
        return valid_tags, invalid_tags

    return [], []


DETECTION_PREFIX = inspect.cleandoc(f"""
        You are a expert TTS emotion cue detector.

        Use the PREVIOUS and NEXT sentences ONLY as context to understand the emotional tone of the TARGET sentence.

        Allowed TAGS: {TTS_TAGS_STR}

        IMPORTANT RULES:
        1. Many TARGET sentences are NEUTRAL and must return {{"tags": []}}.
        2. If the TARGET sentence expresses no emotion, no vocal tension, and no implied emotional state, you MUST return {{"tags": []}}.
        3. Do NOT guess. If unsure, return {{"tags": []}}.
        4. Only detect emotion if it is clearly present INSIDE the TARGET sentence. 
           - Emotion implied ONLY in context (previous/next) does NOT count unless the TARGET sentence continues it.
        5. NEVER output any tag not in Allowed TAGS.
        6. Be precise, literal, and deterministic. Avoid creative interpretation beyond observable cues.

        TASK:
        1. Identify any matching emotions of the TARGET sentence, then select and return the single strongest one.

        OUTPUT FORMAT (STRICT):
            - If 0 tags match → {{"tags": []}}
            - If 1+ tag matches → {{"tags": ["TAG1"]}}

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


def build_dynamic_detection_suffix(prev_s, curr_s, next_s):
    return inspect.cleandoc(f"""
        {prev_s}

        TARGET SENTENCE:
        {curr_s}

        NEXT SENTENCE:
        {next_s}

        OUTPUT:
    """)


VERIFIER_PREFIX = inspect.cleandoc("""
        You are an expert TTS emotion cue verifier. 
        Verify whether a TARGET sentence explicitly expresses a specific TONE emotion.

        Verification Rules:
        1. Confirm the tone ONLY if the TARGET sentence itself clearly expresses that tone
           through wording, phrasing, attitude, or obvious vocal intent.
        2. Emotional clues in the PREVIOUS or NEXT sentences do NOT count unless the TARGET
           sentence directly continues that same emotion.
        3. A QUESTION (anything ending with '?') is NOT a tone such as ANGRY, SARCASTIC, DISAPPOINTED, or MISCHIEVOUS 
           unless the wording clearly shows hostility, mockery, frustration, or playful teasing.
        4. Do NOT guess. If the tone is not clearly present in the TARGET sentence, you MUST respond "NO".
        5. If the tone IS clearly present in the TARGET sentence, output "YES".
        6. Output strictly one word: "YES" or "NO".
        """) + "\n"


def build_dynamic_verification_suffix(prev_s, curr_s, next_s, tag):
    return inspect.cleandoc(f"""
        TONE TAG TO VERIFY: {tag}

        PREVIOUS: {prev_s}
        TARGET: {curr_s}
        NEXT: {next_s}
        
        Does the TARGET sentence clearly express the tone {tag}, YES or NO?
        OUTPUT:
    """)


DETECTION_STATIC_MASK = None
DETECTION_STATIC_BATCH_PAST = None

VERIFICATION_STATIC_MASK = None
VERIFICATION_STATIC_BATCH_PAST = None


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


def init_verification_prefix_cache(model, tokenizer, BATCH_SIZE):
    global VERIFICATION_STATIC_MASK, VERIFICATION_STATIC_BATCH_PAST

    if VERIFICATION_STATIC_MASK is not None and VERIFICATION_STATIC_BATCH_PAST is not None:
        return  # already initialized

    device = next(model.parameters()).device

    enc = tokenizer(VERIFIER_PREFIX, return_tensors="pt").to(device)
    static_ids = enc["input_ids"]  # (1, prefix_len)
    VERIFICATION_STATIC_MASK = enc["attention_mask"]  # (1, prefix_len)

    with torch.inference_mode():
        out = model(
            input_ids=static_ids,
            attention_mask=VERIFICATION_STATIC_MASK,
            use_cache=True,
        )

    # detach so we don't track autograd history
    verification_static_past = tuple((k.detach(), v.detach()) for k, v in out.past_key_values)

    VERIFICATION_STATIC_BATCH_PAST = repeat_past_kv(verification_static_past, BATCH_SIZE)


def detect_batch(title, indices, sentences, model, tokenizer, outputPath, BATCH_SIZE):
    global DETECTION_STATIC_MASK, DETECTION_STATIC_BATCH_PAST
    device = next(model.parameters()).device

    outputs = ["" for _ in range(len(indices))]
    writer = SummaryWriter(log_dir=f"{outputPath}runs/{title}")
    for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"{title}", ncols=90, position=1):
        chunk = indices[i:i + BATCH_SIZE]
        start = time.time()
        dynamic_prompts = []
        for idx in chunk:
            prev_s, curr_s, next_s = get_context(idx, sentences)
            dynamic_prompts.append(build_dynamic_detection_suffix(prev_s, curr_s, next_s))

        dyn = tokenizer(dynamic_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        dyn_ids = dyn["input_ids"]  # (B, dyn_len)
        dyn_mask = dyn["attention_mask"]  # (B, dyn_len)

        batch_size = dyn_ids.size(0)
        # prefix_len = DETECTION_STATIC_IDS.size(1)

        static_mask = DETECTION_STATIC_MASK.repeat(batch_size, 1)  # (B, prefix_len)
        full_mask = torch.cat([static_mask, dyn_mask], dim=1)  # (B, prefix_len + dyn_len)

        # repeat prefix KV across batch

        past_batch_val = DETECTION_STATIC_BATCH_PAST
        if batch_size < BATCH_SIZE:
            past_batch_val = slice_prefix_kv(DETECTION_STATIC_BATCH_PAST, batch_size)

        try:
            out_ids = fast_generate(
                model,
                dynamic_ids=dyn_ids,
                attention_mask=full_mask,
                max_new_tokens=10,
                eos_token_id=tokenizer.eos_token_id,
                past_key_values=past_batch_val,
            )
            batch_outputs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"Exception during detection: {e}. Model cannot detect emotions.")
            batch_outputs = [""] * len(chunk)
        finally:
            del out_ids
            del static_mask
            del full_mask

        verifications = []
        # Parse tags for each sentence
        for local_i, idx in enumerate(chunk):
            try:
                start_pos = batch_outputs[local_i].find("OUTPUT:")
                if start_pos == -1:
                    outputs[indices.index(idx)] = ""
                else:
                    tags, invalid_tags = extract_emotion_tags(batch_outputs[local_i][start_pos:])
                    if tags:
                        if tags[0] in SOUNDS:
                            outputs[indices.index(idx)] = [tags[0]] + invalid_tags
                        else:
                            verifications.append((tags[0], invalid_tags, idx))
                    else:
                        outputs[indices.index(idx)] = invalid_tags
            except Exception as e:
                print(f"Exception while parsing tags: {e}")
                outputs[indices.index(idx)] = ""

        if verifications:
            verified_outputs = tag_verification(model, tokenizer, sentences, BATCH_SIZE, verifications)
            for idx, modified_tags in verified_outputs.items():
                outputs[indices.index(idx)] = modified_tags

        end = time.time()
        writer.add_scalar("EmotionDetection/GenerationTime", end - start, i+1)

        clear_cache()

    writer.flush()
    writer.close()
    return outputs


def tag_verification(model, tokenizer, sentences, BATCH_SIZE, verifications):
    global VERIFICATION_STATIC_MASK, VERIFICATION_STATIC_BATCH_PAST
    dynamic_prompts = []
    for tag, _, idx in verifications:
        prev_s, curr_s, next_s = get_context(idx, sentences)
        dynamic_prompts.append(build_dynamic_verification_suffix(prev_s, curr_s, next_s, tag[1:-1]))

    dyn = tokenizer(dynamic_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    dyn_ids = dyn["input_ids"]
    dyn_mask = dyn["attention_mask"]

    batch_size = dyn_ids.size(0)

    # full attention mask = [prefix_mask, dyn_mask]
    static_mask = VERIFICATION_STATIC_MASK.repeat(batch_size, 1)  # (B, prefix_len)
    full_mask = torch.cat([static_mask, dyn_mask], dim=1)  # (B, prefix_len + dyn_len)

    # repeat prefix KV across batch

    past_batch_val = VERIFICATION_STATIC_BATCH_PAST
    if batch_size < BATCH_SIZE:
        past_batch_val = slice_prefix_kv(VERIFICATION_STATIC_BATCH_PAST, batch_size)

    try:
        out_ids = fast_generate(
            model,
            dynamic_ids=dyn_ids,
            attention_mask=full_mask,
            max_new_tokens=10,
            eos_token_id=tokenizer.eos_token_id,
            past_key_values=past_batch_val,
        )
        batch_outputs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)

    except Exception as e:
        print(f"Exception during verification: {e}. Model cannot verify emotions.")
        batch_outputs = ["No"] * len(verifications)
    finally:
        del out_ids
        del static_mask
        del full_mask

    verified_outputs = {}
    for local_i, (tag, invalid_tag, idx) in enumerate(verifications):
        verified_outputs[idx] = []
        try:
            start_pos = batch_outputs[local_i].find("OUTPUT:")
            if start_pos != -1:
                if "YES" in batch_outputs[local_i][start_pos:].strip().upper():
                    verified_outputs[idx] = [tag]
        except Exception as e:
            print(f"Exception while verifying tags: {e}")
        finally:
            verified_outputs[idx] += invalid_tag

    del batch_outputs

    return verified_outputs


def detectEmotions(model, tokenizer, pages, notebook_name, section_name, EMOTION_CACHE, outputPath):

    progress = 0
    BATCH_SIZE = 20
    print("Starting KV batch prefix caching.")
    init_detection_prefix_cache(model, tokenizer, BATCH_SIZE)
    init_verification_prefix_cache(model, tokenizer, BATCH_SIZE)
    print("Completed KV batch prefix caching.")

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
            output = detect_batch(page["title"], indices, lines, model, tokenizer, outputPath, BATCH_SIZE)
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
                updateCache('cache/emotionCache.json', EMOTION_CACHE)
                progress += 1
            else:
                print(f"Something went wrong while detecting emotions for {page['title']}.")
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}.")

    del DETECTION_STATIC_MASK
    del DETECTION_STATIC_BATCH_PAST
    del VERIFICATION_STATIC_MASK
    del VERIFICATION_STATIC_BATCH_PAST
    clear_cache()

    return progress
