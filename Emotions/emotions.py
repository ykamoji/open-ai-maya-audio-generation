import torch
import os
import pandas as pd
import difflib
import re
import random
import spacy
from tqdm import tqdm
from collections import Counter, deque
from Emotions.utils import getModelAndTokenizer, clean_output, ALLOWED_TAGS, AUTO_CORRECT_MAP, TAG_PENALTY_WEIGHTS, \
    TAG_SIMILARITY_WEIGHTS

nlp = spacy.load("en_core_web_md")

TAG_STATS_PATH = 'emotion_tag_stats.csv'
UNKNOWN_TAG_LOG_PATH = "unknown_tags_log.csv"
unknown_tag_log = Counter()

prompt = """
You are an expressive text enhancer that adds emotional cues to writing.
Your task:
Insert the most appropriate emotion tags **after** specific words or phrases in a paragraph.
Emotion tags you may use: <laugh>, <laugh_harder>, <sigh>, <chuckle>, <gasp>, <angry>, <excited>, <whisper>, <cry>, <scream>, <sing>, <snort>, <exhale>, <gulp>, <giggle>, <sarcastic>, <curious>.
Rules:
1. Keep the original text exactly as written. Do not paraphrase or rewrite.
2. Insert emotion tags only where natural emotional cues are implied.
3. Add the tag **right after** the word or phrase it applies to, before punctuation if it fits better.
4. Use tags sparingly and only when contextually appropriate.
5. Reflect tone (anger, excited, curious, laugh, sarcastic, etc.) based on clues in the text.
Example 1:
Input: "He looked at the strange machine and tilted his head."
Answer: "He looked at the strange machine <curious> and tilted his head."
Example 2:
Input: "I can’t believe you actually did that. You’re unbelievable!"
Answer: "I can’t believe you actually did that. <gasp> You’re unbelievable! <laugh>"
Example 3:
Input: "I tried so hard, but it still wasn’t enough."
Answer: "I tried so hard, <sigh> but it still wasn’t enough <cry>."
Now process the following paragraph and add the most appropriate emotion tags:
"""

if os.path.exists('emotion_tag_stats.csv'):
    df_stats = pd.read_csv(TAG_STATS_PATH)
    global_tag_counts = Counter(dict(zip(df_stats["tag"], df_stats["count"])))
    # print(f"Loaded previous tag stats ({sum(global_tag_counts.values())} total tags).")
else:
    global_tag_counts = Counter({tag: 0 for tag in ALLOWED_TAGS})
    df_stats = pd.DataFrame({"tag": ALLOWED_TAGS, "count": [0] * len(ALLOWED_TAGS)})
    # print("🆕 Initialized new tag frequency tracker.")

HISTORY_WINDOW = 5  # Last 5 paragraphs matter
recent_tag_history = deque(maxlen=HISTORY_WINDOW)


def spacy_similarity(a, b):
    return nlp(a).similarity(nlp(b))


def weighted_fuzzy_match(raw, allowed):
    raw = raw.lower().strip()
    best_score = -1
    best_tag = None

    for tag in allowed:
        fuzzy = difflib.SequenceMatcher(None, raw, tag).ratio()
        semantic = spacy_similarity(raw, tag)
        bonus = 0.0
        for k, related in TAG_SIMILARITY_WEIGHTS.items():
            if k in raw:
                if tag in related:
                    bonus = 0.25
        score = (0.2 * fuzzy) + (0.7 * semantic) + bonus
        if score > best_score:
            best_score = score
            best_tag = tag

    return best_tag


def autocorrect_tag(tag):
    global unknown_tag_log
    t = tag.lower().strip()

    if t not in ALLOWED_TAGS and t not in AUTO_CORRECT_MAP:
        unknown_tag_log[t] += 1

    if t in AUTO_CORRECT_MAP:
        return f"<{AUTO_CORRECT_MAP[t]}>"

    if t in ALLOWED_TAGS:
        return f"<{t}>"

    best = weighted_fuzzy_match(t, ALLOWED_TAGS)

    return f"<{best}>" if best else ""


def remove_and_autocorrect_tags(text):
    return re.sub(r"<([^>]+)>", lambda m: autocorrect_tag(m.group(1)), text)


def extract_tags(text):
    return re.findall(r'<([^>]+)>', text)


def moderate_tag_reuse(text, max_repeat_per_tag=2):
    counts = Counter()
    result = []
    tokens = re.split(r'(<[^>]+>)', text)
    for t in tokens:
        if re.match(r'<([^>]+)>', t):
            tag = re.findall(r'<([^>]+)>', t)[0]
            if tag in ALLOWED_TAGS:
                counts[tag] += 1
                if counts[tag] > max_repeat_per_tag:
                    continue
        result.append(t)
    return "".join(result)


def build_penalty_feedback():
    recent_counts = Counter(recent_tag_history)
    if not recent_counts:
        return ""

    penalty_strength = {}
    for tag, cnt in recent_counts.items():
        weight = TAG_PENALTY_WEIGHTS.get(tag, 1.0)
        penalty_strength[tag] = cnt * weight

    sorted_tags = sorted(penalty_strength.items(), key=lambda x: x[1], reverse=True)
    if not sorted_tags:
        return ""

    mild = [t for t, s in sorted_tags if 2 <= s < 3]
    medium = [t for t, s in sorted_tags if 3 <= s < 5]
    strong = [t for t, s in sorted_tags if s >= 5]

    parts = []
    if mild:
        parts.append("slightly more often recently: " + ", ".join(f"<{t}>" for t in mild))
    if medium:
        parts.append("moderately often recently: " + ", ".join(f"<{t}>" for t in medium))
    if strong:
        parts.append("quite often recently: " + ", ".join(f"<{t}>" for t in strong))

    if not parts:
        return ""

    feedback = (
        "\nNote on recent tag usage: " +
        " ; ".join(parts) +
        ". If multiple tags fit, prefer a different one to increase variety.\n"
    )
    return feedback


def add_global_feedback():
    if random.random() < 0.70:  # 70% of the time: no penalty → prevents monotonic bias
        return prompt + "{}\nAnswer:"
    feedback = build_penalty_feedback()
    return (prompt + feedback if feedback else prompt) + "{}\nAnswer:"


def save_global_stats():
    df = (pd.DataFrame.from_dict(global_tag_counts, orient="index", columns=["count"])
        .reset_index()
        .rename(columns={"index": "tag"})
    )
    df = df.sort_values("count", ascending=False)
    df.to_csv(TAG_STATS_PATH, index=False)


def save_unknown_tag_log():
    if not unknown_tag_log:
        return

    df = pd.DataFrame(
        [(tag, count) for tag, count in unknown_tag_log.items()],
        columns=["unknown_tag", "count"]
    )

    if os.path.exists(UNKNOWN_TAG_LOG_PATH):
        try:
            existing = pd.read_csv(UNKNOWN_TAG_LOG_PATH)
            merged = (
                pd.concat([existing, df])
                .groupby("unknown_tag", as_index=False)
                .sum()
            )
        except Exception:
            merged = df
    else:
        merged = df

    merged = merged.sort_values("count", ascending=False)
    merged.to_csv(UNKNOWN_TAG_LOG_PATH, index=False)


def generate_emotion_lines(model, tokenizer, paragraph):
    global global_tag_counts
    tagged = []

    batch_prompts = add_global_feedback()

    batch_prompts = batch_prompts.replace("{}", paragraph)

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(
        model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    text = [d.strip() for d in decoded]
    text = clean_output(text, [prompt])[0]
    clean_text = moderate_tag_reuse(text)
    clean_text = remove_and_autocorrect_tags(clean_text)
    tags = extract_tags(clean_text)
    for t in tags:
        if t in ALLOWED_TAGS:
            global_tag_counts[t] += 1
            recent_tag_history.append(t)
    tagged.append(clean_text)
    return clean_text


def addEmotions(Args, pages):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize)

    for t in ALLOWED_TAGS:
        TAG_PENALTY_WEIGHTS.setdefault(t, 1.0)

    response = []
    for page in pages:
        content = page['content']
        outputs = []
        for paragraph in tqdm(content, desc="Processing", ncols=100):
            emotion_lines = generate_emotion_lines(model, tokenizer, paragraph)
            outputs.append(emotion_lines)
            torch.cuda.empty_cache()
        response.append({"title": page['title'], "content": outputs})
        save_global_stats()
        save_unknown_tag_log()

    return response
