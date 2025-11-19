import torch
import os
import re
import json
import logging
import pandas as pd
from tqdm import tqdm
from collections import Counter, deque
from Emotions.toneReRanker import rerank
from Emotions.utils import getModelAndTokenizer, split_sentences
from utils import updateCache

if os.path.isfile('emotions.log'):
    os.remove('emotions.log')

logging.basicConfig(
    filename='emotions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

TAG_STATS_PATH = 'emotion_stats.csv'
UNKNOWN_STATS_PATH = "unknown_stats.csv"
unknown_tag_log = Counter()

TONES = [
    "[ANGRY]", "[EXCITED]", "[SARCASTIC]", "[CURIOUS]", "[SING]", "[APPALLED]", "[MISCHIEVOUS]", "[DISAPPOINTED]"
]

SOUNDS = [
    "[LAUGH]", "[LAUGH_HARDER]", "[SIGH]", "[CHUCKLE]", "[GASP]", "[CRY]", "[SCREAM]", "[WHISPER]", "[SNORT]",
    "[EXHALE]", "[GULP]", "[GIGGLE]"
]

TONES_SET = set(TONES)

TTS_TAGS = TONES + SOUNDS

TTS_TAGS_SET = set(TTS_TAGS)

TTS_TAGS_STR = ", ".join(TTS_TAGS)

USE_LLAMA_CPP = True

if os.path.exists(TAG_STATS_PATH):
    df_stats = pd.read_csv(TAG_STATS_PATH)
    global_tag_counts = Counter(dict(zip(df_stats["emotion"], df_stats["count"])))
else:
    tag_tracker = {
        tag.replace("[", "").replace("]", "").lower(): 0
        for tag in TTS_TAGS
    }
    global_tag_counts = Counter(tag_tracker)

HISTORY_WINDOW = 4
recent_tag_history = deque(maxlen=HISTORY_WINDOW)


def penalty_tag_check(tag):
    recent_counts = Counter(recent_tag_history)
    if not recent_counts:
        return True

    penalty_strength = {}
    for t, cnt in recent_counts.items():
        penalty_strength[t] = cnt

    sorted_tags = sorted(penalty_strength.items(), key=lambda x: x[1], reverse=True)

    high_frequency = [t for t, s in sorted_tags if s > 2]

    if tag in high_frequency:
        return False

    return True


def normalize_tags(text):

    def repl(match):
        tag = match.group(1).lower()

        global_tag_counts[tag] += 1
        recent_tag_history.append(tag)

        if len(text.split()) < 5 or penalty_tag_check(tag):
            return f"<{tag}>"
        else:
            logger.warning(f"{tag} is being used very frequently, not inserting in \"{text}\"")
            return ""

    return re.sub(r"\[([A-Za-z0-9_]+)\]", repl, text)


def save_global_stats():
    df = (pd.DataFrame.from_dict(global_tag_counts, orient="index", columns=["count"])
          .reset_index()
          .rename(columns={"index": "emotion"})
          )
    df = df.sort_values("count", ascending=False)
    df.to_csv(TAG_STATS_PATH, index=False)


def save_unknown_tag_log():
    if not unknown_tag_log:
        return

    df = pd.DataFrame(
        [(tag, count) for tag, count in unknown_tag_log.items()],
        columns=["unknown_emotion", "count"]
    )

    if os.path.exists(UNKNOWN_STATS_PATH):
        try:
            existing = pd.read_csv(UNKNOWN_STATS_PATH)
            merged = (
                pd.concat([existing, df])
                .groupby("unknown_emotion", as_index=False)
                .sum()
            )
        except Exception:
            merged = df
    else:
        merged = df

    merged = merged.sort_values("count", ascending=False)
    merged.to_csv(UNKNOWN_STATS_PATH, index=False)


def extract_emotion_tags(text):
    start_pos = text.find("Answer:")
    if start_pos == -1:
        return []
    output = text[start_pos:]
    match = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', output)
    if match:
        inner = match.group(1)
        tags = str(re.findall(r'"([^"]+)"', inner)).replace("'", '"')
        # return [f"[{tag.upper()}]" for tag in json.loads(tags) if f"[{tag.upper()}]" in TTS_TAGS]
        valid_tags = []
        for tag in json.loads(tags):
            t = f"[{tag.upper()}]"
            if t in TTS_TAGS_SET:
                valid_tags.append(t)
            else:
                unknown_tag_log[tag.upper()] += 1
        return valid_tags

    return []


def get_context(i, sentences):
    prev_s = sentences[i - 1] if i - 1 >= 0 else ""
    curr_s = sentences[i]
    next_s = sentences[i + 1] if i + 1 < len(sentences) else ""
    return prev_s, curr_s, next_s


def build_detection_prompt(prev_s, curr_s, next_s):
    return f"""
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
        {prev_s}

        TARGET SENTENCE:
        {curr_s}

        NEXT SENTENCE:
        {next_s}

        Answer:
        """


def detect_and_rank_batch(indices, sentences, model, tokenizer):

    outputs = [[]] * len(indices)

    for start in range(0, len(indices), 2):
        chunk = indices[start:start + 2]
        prompts = []
        for idx in chunk:
            prev_s, curr_s, next_s = get_context(idx, sentences)
            prompts.append(build_detection_prompt(prev_s, curr_s, next_s))

        lines = "\n".join([sentences[idx] for idx in chunk])
        logger.debug(f"Running emotion detection for \"{lines}\"")
        if USE_LLAMA_CPP:
            batch_results = []
            for prompt in prompts:
                try:
                    result = model(
                        prompt,
                        max_tokens=50,
                        temperature=0.0,
                        top_k=1,
                        top_p=1.0,
                        repeat_penalty=1.0,
                    )
                    text = "Answer: " + result["choices"][0]["text"]
                except Exception as e:
                    logger.error(f"Exception: {e}. Model cannot detect emotions.")
                    text = "Answer: "
                batch_results.append(text)
        else:
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        temperature=0.0,
                    )
                batch_results = tokenizer.batch_decode(out, skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Exception: {e}. Model cannot detect emotions.")
                batch_results = []
            finally:
                for var in ["out", "inputs"]:
                    if var in locals():
                        del locals()[var]

        for local_i, idx in enumerate(chunk):
            try:
                base_tags = extract_emotion_tags(batch_results[local_i])
                logger.debug(f"Model returned {base_tags} emotions for \"{sentences[idx]}\"")
                candidate_tags = [
                    t.replace("[", "").replace("]", "").lower()
                    for t in base_tags
                ]
                updated, scores = rerank(sentences[idx], candidate_tags, genre="YA", top_k=2)
                if len(base_tags) > 0:
                    if len(candidate_tags) == 0 or base_tags[0] != candidate_tags[0]:
                        logger.warning(f"Re ranking updated from {base_tags} to {updated} ({scores}) emotions for \"{sentences[idx]}\"")
                outputs[indices.index(idx)] = updated
            except Exception as e:
                logger.error(f"Exception in rerank batch: {e}")
                outputs[indices.index(idx)] = []

    return outputs


def index_sentence(sentence: str):
    words = sentence.split()
    return " | ".join(f"{i}:{w}" for i, w in enumerate(words))


def extract_index(text: str):
    start_pos = text.find("Answer:")
    if start_pos == -1:
        return None
    output = text[text.find("Answer:") - 50:]
    m = re.search(r'\b(END|\d+)\b', output)
    if m:
        val = m.group(1)
        return int(val) if val.isdigit() else val
    return None


def safe_insert(sentence: str, index, tag: str):
    words = sentence.split()
    if index == "END" or index >= len(words):
        return " ".join(words) + f" {tag}"
    return " ".join(words[:index] + [f"{tag}"] + words[index:])


def shift_emotion_inside(sentence):
    # Pattern:
    #   (1) capture everything before final punctuation
    #   (2) capture final punctuation (., ?, !, optionally followed by a quote)
    #   (3) capture the emotion tag at the end: [ANYTHING]
    pattern = re.compile(r'^(.*?)([\.!?]["\']?)\s*(\[[A-Za-z0-9_]+\])\s*$')

    m = pattern.match(sentence)
    if m:
        before, punctuation, tag = m.groups()
        sentence = f"{before} {tag}{punctuation}"

    return sentence


def build_placement_prompt(sentence: str, tag: list):
    prompt = f"""
        You are an expert at placing TTS emotion tags into sentences.

        Your job is to choose the BEST position to insert the tag {tag} inside the TARGET sentence, 
        without modifying any words. The tag will be inserted BEFORE the chosen word index.

        RULES:
        1. If the emotional cue is direct or indirect 
           (examples: wide-eyed, shocked, tense, uneasy, startled, frozen, nervous, hesitant, trembling),
           place the tag at the strongest emotional point of the sentence.
        2. You may place the tag either BEFORE or AFTER the emotional cue word. 
           Choose whichever placement gives the clearest expressive delivery.
        3. If the emotional moment resolves at the end of the sentence, return END.
        4. Return ONLY one of the following:
           - A single integer index (0-based)
           - END  (if the best placement is at the end of the sentence)
        5. Do NOT output anything else.
        6. Do NOT rewrite the sentence.
        7. Do NOT output explanations.

        EXAMPLE 1:
        TAG: EXCITED
        TARGET SENTENCE:
        A spark of anticipation flickered through her as she stepped toward the doorway.
        BEST INSERTION INDEX:
        5

        EXAMPLE 2:
        TAG: EXHALE
        TARGET SENTENCE:
        He paused for a moment, a nervous tremor in his voice before he continued speaking.
        BEST INSERTION INDEX:
        11

        EXAMPLE 3:
        TAG: SIGH
        TARGET SENTENCE:
        She lowered her shoulders as the weight of the moment settled heavily on her.
        BEST INSERTION INDEX:
        END

        TARGET SENTENCE:
        {sentence}

        Answer:
        """

    return prompt


def insert_emotion_tag(sentences, tags, model, tokenizer):
    prompts = [build_placement_prompt(s, t) for s, t in zip(sentences, tags)]
    modified_sentences = sentences[:]
    try:
        if USE_LLAMA_CPP:
            batch_decoded = []
            for prompt in prompts:
                try:
                    result = model(
                        prompt,
                        max_tokens=10,
                        temperature=0.0,
                        stop=[]
                    )
                    batch_decoded.append("Answer: " + result["choices"][0]["text"].strip())
                except Exception as e:
                    logger.error(f"Exception during placement call: {e}.")
                    batch_decoded.append("")
        else:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to('cuda')

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )

            batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, decoded in enumerate(batch_decoded):
            index = extract_index(decoded)
            if index is None:
                logger.error(f"Model couldn't find the index in the output \"{decoded}\"\nDefaulting to END.")
                index = "END"
            else:
                is_valid_int = isinstance(index, int) and (0 <= index < len(sentences[i]))
                index = index if (index == "END" or is_valid_int) else "END"
            modified_sentences[i] = safe_insert(sentences[i], index, tags[i])

    except Exception as e:
        logger.error(f"Exception: {e}. Not inserting emotion to this batch.")
    finally:
        for var in ["batch_decoded", "output", "inputs"]:
            if var in locals():
                del locals()[var]

    return modified_sentences


DEFAULT_PRESETS = set(TONES + ['[LAUGH_HARDER]'])


def process_detection(paragraph, model, tokenizer):
    sentences = split_sentences(paragraph)
    detections = []
    modified_lines = {}

    indices = list(range(len(sentences)))

    batch_tags = detect_and_rank_batch(indices, sentences, model, tokenizer)

    for i, s in enumerate(sentences):
        try:
            tags = batch_tags[i]
            if tags:
                if tags[0] not in DEFAULT_PRESETS:
                    detections.append((i, s, tags[0]))
                else:
                    logger.debug(f"Running custom rules on \"{s}\" for {tags[0]}")
                    modified_lines[i] = process_tone_rules(s, tags[0])
                    logger.debug(f"Running custom rules on \"{s}\" for {tags[0]}: \"{modified_lines[i]}\"")
            else:
                logger.debug(f"No emotion detected for \"{s}\"")
                modified_lines[i] = s
        except Exception as e:
            logger.error(f"Exception: {e}. Skipping emotion detection on \"{sentences[i]}\"")

    return modified_lines, detections


def process_tone_rules(sentence, tag):
    if tag in TONES_SET:
        if sentence.startswith("\""):
            return f"\"{tag} {sentence[1:]}"
        return f"{tag} {sentence}"

    if tag == '[LAUGH_HARDER]':
        return f"{sentence.strip('.')} {tag}."


def process_paragraph(paragraph, model, tokenizer):
    modified_lines, detections = process_detection(paragraph, model, tokenizer)
    sentences = []
    tags = []
    insert_line_pos = []
    for (i, s, t) in detections:
        sentences.append(s)
        tags.append(t)
        insert_line_pos.append(i)

    if sentences:
        modified_sentences = insert_emotion_tag(sentences, tags, model, tokenizer)
        logger.debug(f"Completed emotion tag placements batch.")
        for idx, m_s in enumerate(modified_sentences):
            modified_lines[insert_line_pos[idx]] = m_s

    ## Apply post-processing for tag at the end
    emotion_lines = []
    keys = sorted(modified_lines.keys())
    for key in keys:
        emotion_line = shift_emotion_inside(modified_lines[key])
        emotion_line = normalize_tags(emotion_line)
        emotion_line = re.sub(r"\s+", " ", emotion_line).strip()
        emotion_lines.append(emotion_line)

    recent_tag_history.clear()
    return emotion_lines + [" "]


def addEmotions(Args, pages, EMOTION_CACHE):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

    global global_tag_counts

    progress = 0
    for page in tqdm(pages, desc="Pages", ncols=100, position=0):
        logger.info(f"Starting {page['title']}...")
        content = page['content']
        outputs = []
        try:
            for paragraph in tqdm(content, desc="Paragraphs", ncols=90, position=1):
                outputs.extend(process_paragraph(paragraph, model, tokenizer))
            if outputs:
                outputs.insert(0, page['suggested_title'])
                EMOTION_CACHE[page["title"]] = outputs
                updateCache('emotionCache.json', EMOTION_CACHE)
                progress += 1
                logger.info(f"Completed {page['title']}.")
            else:
                logger.error(f"Something went wrong creating emotions for {page['title']}.")
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}.")
        finally:
            torch.cuda.empty_cache()
            save_global_stats()
            save_unknown_tag_log()

    return progress
