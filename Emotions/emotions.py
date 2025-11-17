import torch
import os
import re
import json
import logging
import pandas as pd
from tqdm import tqdm
from collections import Counter, deque
from Emotions.toneReRanker import strict_rerank
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
    "[ANGRY]", "[EXCITED]", "[SARCASTIC]", "[WHISPER]", "[CURIOUS]", "[SCREAM]", "[SING]"
]

SOUNDS = [
    "[LAUGH]", "[LAUGH_HARDER]", "[SIGH]", "[CHUCKLE]", "[GASP]", "[CRY]", "[SNORT]", "[EXHALE]", "[GULP]", "[GIGGLE]"
]

TONES_SET = set(TONES)

TTS_TAGS = TONES + SOUNDS

TTS_TAGS_SET = set(TTS_TAGS)

TTS_TAGS_STR = ", ".join(TTS_TAGS)


if os.path.exists(TAG_STATS_PATH):
    df_stats = pd.read_csv(TAG_STATS_PATH)
    global_tag_counts = Counter(dict(zip(df_stats["emotion"], df_stats["count"])))
else:
    tag_tracker = {
        tag.replace("[", "").replace("]", "").lower() : 0
        for tag in TTS_TAGS
    }
    global_tag_counts = Counter(tag_tracker)

HISTORY_WINDOW = 50
recent_tag_history = deque(maxlen=HISTORY_WINDOW)


def penalty_tag_check(tag):

    recent_counts = Counter(recent_tag_history)
    if not recent_counts:
        return True

    penalty_strength = {}
    for tag, cnt in recent_counts.items():
        penalty_strength[tag] = cnt

    sorted_tags = sorted(penalty_strength.items(), key=lambda x: x[1], reverse=True)

    high_frequency = [t for t, s in sorted_tags if s > 10]

    if tag in high_frequency:
        return False

    return True


def normalize_tags(text):
    def repl(match):
        tag = match.group(1).lower()

        global_tag_counts[tag] += 1
        recent_tag_history.append(tag)

        if penalty_tag_check(tag):
            return f"<{tag}>"
        else:
            logger.warning(f"{tag} is being used very frequently, not inserting in {text}")
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


def extract_emotion_tags(text: str):
    output = text[text.find("Answer:"):]
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
        3. Return ONLY the top three tags (0–3 tags).

        OUTPUT FORMAT (STRICT):
            - If 0 tags match → {{"tags": []}}
            - If 1 tag matches → {{"tags": ["TAG1"]}}
            - If 2 tags match → {{"tags": ["TAG1", "TAG2"]}}
            - If 3+ tags match → {{"tags": ["TAG1", "TAG2", "TAG3"]}}


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


def detect_and_rank_with_context(i, sentences, model, tokenizer):
    prev_s, curr_s, next_s = get_context(i, sentences)

    logger.info(f"Running emotion detection for \"{curr_s}\"")
    prompt = build_detection_prompt(prev_s, curr_s, next_s)

    tags = []
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()

        tags = extract_emotion_tags(decoded)

        logger.info(f"Model returned {tags} emotions for \"{curr_s}\".")

        # try:
            # Re ranking. Available genre: normal, YA, fantasy, drama
            # candidate_tags = [tag.replace('[','').replace(']','').lower() for tag in tags]
            # tags, score = strict_rerank(curr_s, prev_s, next_s, candidate_tags, genre='YA', top_k=3)
            # logger.info(f"Re ranking updated {tags} ({score}) emotions for \"{curr_s}\".")
        # except Exception as e:
            # logger.error(f"Re ranking exception: {e}. \"{curr_s}\" \"{prev_s}\" \"{next_s}\" {tags}")

    except Exception as e:
        logger.error(f"Exception: {e}. Model not returning any emotion for \"{curr_s}\"")

    finally:
        for var in ["decoded", "output", "inputs"]:
            if var in locals():
                del locals()[var]

    logger.info(f"Completed emotion {tags} for \"{curr_s}\"")
    return tags


def index_sentence(sentence: str):
    words = sentence.split()
    return " | ".join(f"{i}:{w}" for i, w in enumerate(words))


def extract_index(text: str):
    output = text[text.find("Answer:") - 50:]
    m = re.search(r'\d+', output)
    return int(m.group()) if m else None


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
    if not m:
        return sentence  # Nothing to fix

    before, punctuation, tag = m.groups()

    return f"{before} {tag}{punctuation}"


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
            index = index if index else "END"
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
    try:
        for i, s in enumerate(sentences):
            tags = detect_and_rank_with_context(i, sentences, model, tokenizer)
            if tags:
                if tags[0] not in DEFAULT_PRESETS:
                    detections.append((i, s, tags[0]))
                else:
                    logger.info(f"Running custom rules on \"{s}\" for {tags[0]}")
                    modified_lines[i] = process_emotion_rules(s, tags[0])
                    logger.info(f"Running custom rules on \"{s}\" for {tags[0]}: \"{modified_lines[i]}\"")
            else:
                logger.info(f"No emotion detected for \"{s}\"")
                modified_lines[i] = s
    except Exception as e:(
        logger.error(f"Exception: {e}. Skipping detecting emotion on this batch."))

    return modified_lines, detections


def process_emotion_rules(sentence, tag):

    if tag in TONES_SET:
        return f"{tag} {sentence}"

    if tag == '[LAUGH_HARDER]':
        return f"{sentence.strip('.')} {tag}."

    if tag == '[LAUGH]':
        target = r"\[LAUGH\]"

        # Pattern to detect [LAUGH] at end with optional preceding/trailing punctuation
        end_pattern = re.compile(rf"[\.\!\?]?\s*{target}[\.\!\?]?\s*$")

        # If [LAUGH] is effectively at the end, don't modify
        if end_pattern.search(sentence):
            return sentence

        return sentence.replace("[LAUGH]", "-[LAUGH]-")


def process_paragraph(paragraph, model, tokenizer):

    modified_lines, detections = process_detection(paragraph, model, tokenizer)
    sentences = []
    tags = []
    insert_line_pos = []
    for (i, s, t) in detections:
        sentences.append(s)
        tags.append(t)
        insert_line_pos.append(i)

    logger.info(f"Running emotion tag placements.")
    modified_sentences = insert_emotion_tag(sentences, tags, model, tokenizer)
    logger.info(f"Completed emotion tag placements.")
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

    return emotion_lines


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
                outputs.insert(0, pages['suggested_title'])
                EMOTION_CACHE[page["title"]] = outputs
                updateCache('emotionCache.json', EMOTION_CACHE)
                progress += 1
                logger.info(f"Completed emotions for {page['title']}.")
            else:
                logger.error(f"Something went wrong creating emotions for {page['title']}.")
        except Exception as e:
            print(f"Exception: {e}. Skipping {page['title']}.")
        finally:
            torch.cuda.empty_cache()
            save_global_stats()
            save_unknown_tag_log()

    return progress
