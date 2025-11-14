import torch
import os
import pandas as pd
import re
from tqdm import tqdm
from collections import Counter, deque
from Emotions.utils import getModelAndTokenizer, ALLOWED_TAGS, AUTO_CORRECT_MAP, TAG_PENALTY_WEIGHTS


TAG_STATS_PATH = 'emotion_tag_stats.csv'
UNKNOWN_TAG_LOG_PATH = "unknown_tags_log.csv"
unknown_tag_log = Counter()

BASE_SYSTEM_PROMPT = """
You are a strict text-tagging assistant.

YOUR ONLY TASK:
Insert emotion/action TAGS into the user’s paragraph.

HARD RULES — FOLLOW EXACTLY:
1. Do NOT change any original words.
2. Do NOT add new words.
3. Do NOT delete any words.
4. Do NOT reorder any words.
5. Insert ONLY tags.
6. Tags must be UPPERCASE and in square brackets: [LAUGH]
7. Insert the tag IMMEDIATELY BEFORE the matching word or phrase.
8. If a trigger does NOT match exactly → add NO TAG.
9. Output ONLY the tagged paragraph. No explanations.

ALLOWED TAGS:
[TAGS]

TRIGGER → TAG (exact match only):
[TRIGGER]

DO NOT TAG:
- Emotion adjectives: sad, angry, scared
- Tone adverbs: sadly, angrily, nervously, happily
- Any action NOT listed above
- Partial or fuzzy matches

EXAMPLES (follow format exactly):
[EXAMPLES]

FINAL REQUIREMENT:
Return ONLY the paragraph with inserted tags.
Nothing else.
"""

TRIGGER = [
    """"- gasped", "breath caught" → [GASP]""",
    """"- whispered", "voice barely audible" → [WHISPER]""",
    """"- tears fell", "crying", "started crying" → [CRY]""",
    """"- smirk", "eye-roll", "rolled her eyes" → [SARCASTIC]""",
    """"- eyes widened", "wide-eyed" → [CURIOUS]""",
]

EXAMPLES = [
    """He suddenly [GASP] gasped.""",
    """Her [WHISPER] voice barely audible, she spoke.""",
    """His [CURIOUS] eyes widened at the news.""",
    """She gave a [SARCASTIC] smirk.""",
    """He [CRY] started crying.""",
]


if os.path.exists('emotion_tag_stats.csv'):
    df_stats = pd.read_csv(TAG_STATS_PATH)
    global_tag_counts = Counter(dict(zip(df_stats["tag"], df_stats["count"])))
else:
    global_tag_counts = Counter({tag: 0 for tag in ALLOWED_TAGS})

HISTORY_WINDOW = 5  # Last 5 paragraphs matter
recent_tag_history = deque(maxlen=HISTORY_WINDOW)


def autocorrect_tag(tag):
    t = tag.lower().strip()

    if t in ALLOWED_TAGS:
        return f"<{t}>"

    if t in AUTO_CORRECT_MAP:
        return f"<{AUTO_CORRECT_MAP[t]}>"

    unknown_tag_log[t] += 1
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


def normalize_tags(text):
    def repl(match):
        tag = match.group(1).lower()
        return f"<{tag}>"

    return re.sub(r"\[([A-Za-z0-9_]+)\]", repl, text)


def build_penalty_tags():
    recent_counts = Counter(recent_tag_history)
    if not recent_counts:
        return ALLOWED_TAGS, TRIGGER, EXAMPLES

    penalty_strength = {}
    for tag, cnt in recent_counts.items():
        weight = TAG_PENALTY_WEIGHTS.get(tag, 1.0)
        penalty_strength[tag] = cnt * weight

    sorted_tags = sorted(penalty_strength.items(), key=lambda x: x[1], reverse=True)

    if not sorted_tags:
        return ALLOWED_TAGS, TRIGGER, EXAMPLES

    high = [t for t, s in sorted_tags if s > HISTORY_WINDOW]

    if not high:
        return ALLOWED_TAGS, TRIGGER, EXAMPLES
    else:
        NEW_ALLOWED_TAGS = [tag for tag in ALLOWED_TAGS if tag.lower() not in [h.lower() for h in high]]

        def extract_tag(text):
            match = re.search(r'\[([A-Z_]+)\]', text)
            return match.group(1).lower() if match else None

        NEW_TRIGGERS = []
        for t in TRIGGER:
            tag = extract_tag(t)
            if tag and tag in NEW_ALLOWED_TAGS:
                NEW_TRIGGERS.append(t)

        NEW_EXAMPLES = []
        for e in EXAMPLES:
            tag = extract_tag(e)
            if tag and tag in NEW_ALLOWED_TAGS:
                NEW_EXAMPLES.append(e)

        return NEW_ALLOWED_TAGS, NEW_TRIGGERS, NEW_EXAMPLES


def build_system_prompt():
    tags, triggers, examples = build_penalty_tags()
    tags = ", ".join(f"[{t.upper()}]" for t in tags)
    triggers = "\n".join(f"[{t.upper()}]" for t in triggers)
    examples = "\n".join(f"[{t.upper()}]" for t in examples)

    return BASE_SYSTEM_PROMPT.replace("[TAGS]", tags).replace("[TRIGGER]", triggers).replace("[EXAMPLES]", examples)


def make_user_prompt(paragraph: str) -> str:
    return f"""
    Insert tags into the paragraph. Do NOT change, remove, or add any words.
    Paragraph:
    {paragraph}
    Return ONLY the tagged paragraph."""


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

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": make_user_prompt(paragraph)}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.0,  # deterministic
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0]
    new_tokens = generated_ids[input_ids.shape[-1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    del input_ids, outputs, generated_ids

    completion = normalize_tags(completion)
    clean_text = remove_and_autocorrect_tags(completion)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = moderate_tag_reuse(clean_text)
    tags = extract_tags(clean_text)
    for t in tags:
        if t in ALLOWED_TAGS:
            global_tag_counts[t] += 1
            recent_tag_history.append(t)
    return clean_text


def addEmotions(Args, pages):
    MODEL_PATH = Args.Emotions.ModelPath.__dict__[Args.Platform]

    model, tokenizer = getModelAndTokenizer(MODEL_PATH, Args.Emotions.Quantize, Args.Platform)

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
