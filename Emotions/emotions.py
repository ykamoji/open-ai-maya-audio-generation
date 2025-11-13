import torch
import os
import pandas as pd
import difflib
import re
import random
from tqdm import tqdm
from collections import Counter, deque
from Emotions.utils import getModelAndTokenizer, clean_output, ALLOWED_TAGS, AUTO_CORRECT_MAP, TAG_PENALTY_WEIGHTS

TAG_STATS_PATH = 'emotion_tag_stats.csv'

prompt = """You are an emotion annotator. 
Allowed tags:
<laugh>, <laugh_harder>, <sigh>, <chuckle>, <gasp>, <angry>, <excited>, <whisper>, <cry>, <scream>, <sing>, <snort>, <exhale>, <gulp>, <giggle>, <sarcastic>, <curious>

Your task: 
Insert the above allowed emotion tags after words or short phrases in the paragraph below.

Rules:
1. Keep the original text exactly; do not rewrite or summarize.
2. Add emotion tags only where a clear emotional cue exists.
3. Neutral or factual sentences usually need no tags.
4. Avoid overusing the same tag unless strongly justified.
5. You may add multiple tags per paragraph when emotions shift naturally.
6. Prefer variety of emotion tags within and across paragraphs.
7. Never repeat a tag back-to-back.
8. Do not invent new tags; only use the allowed set.
 
Examples:

Input:
He looked around the empty hall. "So this is it," he said softly.

Answer:
He looked around the empty hall. <exhale> "So this is it," he said softly. <sigh>

---

Input:
“Oh great, another deadline,” she muttered as her laptop froze again.

Answer:
“Oh great, another deadline,” she muttered as her laptop froze again. <sarcastic> <angry>

---

Now tag this paragraph:
{paragraph}
Answer:
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


def autocorrect_tag(tag):
    t = tag.lower().strip()

    if t in AUTO_CORRECT_MAP:
        return f"<{AUTO_CORRECT_MAP[t]}>"

    if t in ALLOWED_TAGS:
        return f"<{t}>"

    close = difflib.get_close_matches(t, ALLOWED_TAGS, n=1, cutoff=0.65)
    if close:
        return f"<{close[0]}>"

    return ""


def remove_and_autocorrect_tags(text):
    return re.sub(r"<([^>]+)>", lambda m: autocorrect_tag(m.group(1)), text)


def extract_tags(text):
    return re.findall(r'<([^>]+)>', text)


def moderate_tag_reuse(text, max_repeat_per_tag=4):
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


def add_global_feedback(prompt):
    if random.random() < 0.70:  # 70% of the time: no penalty → prevents monotonic bias
        return prompt
    feedback = build_penalty_feedback()
    return prompt + feedback if feedback else prompt


def save_global_stats():
    df = pd.DataFrame.from_dict(global_tag_counts, orient="index", columns=["count"]).reset_index()
    df.rename(columns={"index": "tag"}, inplace=True)
    df.to_csv(TAG_STATS_PATH, index=False)


def generate_emotion_lines(model, tokenizer, paragraph):
    batch_prompts = prompt.replace("{}", paragraph)
    global global_tag_counts
    tagged = []

    batch_prompts = add_global_feedback(batch_prompts)

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(
        model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.0,
            do_sample=False,
            repetition_penalty=1.6,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    text = [d.strip() for d in decoded]
    text = clean_output(text, [prompt])
    clean_text = moderate_tag_reuse(text)
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

    return response
