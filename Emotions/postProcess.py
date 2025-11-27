import re
import json
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, deque
from Emotions.emotionDetection import TTS_TAGS, TONES
from tqdm import tqdm
from pathlib import Path
from Emotions.utils import TTS_TAGS_SET


#------------------------------------------------------------------------------------
#  Logic for post processing after stylization.
#  Removing assistant data and splitting up paragraphs
#------------------------------------------------------------------------------------

model_keywords = [
    "Edited paragraph:",
    " (Note:"
]

def voice_post_process(voice_cache):
    post_process_paragraphs = {}
    clean_paragraphs_count = 0
    split_paragraphs_count = 0
    for key in tqdm(voice_cache, desc=f"Page"):
        split_paragraph = False
        cleaned_paragraphs = []
        for paragraph in voice_cache[key]:
            # Remove the prefix at the beginning.
            # for prefix in ["Here's the edited paragraph:\n\n", "Here's the revised paragraph:\n\n"]:
            #     paragraph = paragraph.removeprefix(prefix)

            # Remove the extra details at the end.
            paragraph = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', paragraph)
            if "\n\n" in paragraph:
                parts = paragraph.split("\n\n")
                for i, p in enumerate(parts):
                    if p.strip().startswith("Note:") or p.strip().startswith("I made") or p.strip().startswith(
                            "Changes made:"):
                        parts = parts[:i]
                        clean_paragraphs_count += 1
                        break
                paragraph = "\n\n".join([part.strip() for part in parts])

            if "\n\n" in paragraph:
                split_paragraph = True
                split_paragraphs_count += 1

            for clean_key in model_keywords:
                if clean_key in paragraph:
                    paragraph = paragraph.split(clean_key)[0].strip()

            cleaned_paragraphs.append(paragraph)

        # Keep the list paragraph seperated,
        if split_paragraph:
            final_paragraphs = []
            for p in cleaned_paragraphs:
                for block in p.split("\n\n"):
                    block = block.strip()
                    if block:
                        final_paragraphs.append(block)
            cleaned_paragraphs = final_paragraphs

        post_process_paragraphs[key] = cleaned_paragraphs

    print(f"{clean_paragraphs_count} paragraphs cleaned, {split_paragraphs_count} new paragraphs splits.")

    return post_process_paragraphs


#------------------------------------------------------------------------------------
#  Logic for post processing after emotion detection.
#  Extract tags from output.
#  Tracking unknown tags and window history for tags.
#  Inserting TONE emotion to lines and rest will be insert jobs for next step.
#------------------------------------------------------------------------------------


TAG_STATS_PATH = lambda name: f'cache/stats/{name}'
unknown_tag_log = Counter()

DEFAULT_PRESETS = set(TONES + ['[LAUGH_HARDER]'])
TONES_SET = set(TONES)

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

    if tag in high_frequency or (len(recent_tag_history) > 1 and recent_tag_history[-2] == tag):
        return True

    return False


def process_emotions(L):
    line = L['line']
    if line == "":
        return {
            "line": "",
            "tag": "",
            "invalid_tag": [],
            "messages": []
        }

    output = L['output']
    tags = []
    invalid_tags = []
    for tag in output:
        if tag in TTS_TAGS_SET:
            tags.append(tag)
        else:
            invalid_tags.append(tag)

    messages = []
    if tags:
        messages += [f"Model returned {tags} emotions for \"{line}\""]

    best = tags[0] if tags else None

    processed = {
        "line": L["line"],
        "invalid_tags": invalid_tags,
        "messages": []
    }

    if best:
        processed["tag"] = best
    else:
        if not tags:
            messages += [f"No emotion detected for \"{line}\""]

    processed['line'] = line
    processed['messages'] = messages

    return processed


def process_tone_rules(sentence, tag):
    if tag in TONES_SET:
        if sentence.startswith("\""):
            return f"\"{tag} {sentence[1:]}"
        return f"{tag} {sentence}"

    if tag == '[LAUGH_HARDER]':
        return f"{sentence.strip('.')} {tag}."


################################# M A I N ############################################
def emotion_det_post_process(lines, title):
    processed_results = [None] * len(lines)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_emotions, l): i for i, l in enumerate(lines)}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"{title}"):
            i = futures[f]
            processed_results[i] = f.result()

    cache = []

    for P in processed_results:
        if P['line'] != "" and 'tag' in P and P['tag']:
            line = P['line']
            tag = P["tag"]
            t = tag[1:-1].lower()

            global_tag_counts[t] += 1
            recent_tag_history.append(t)

            if len(line.split()) >= 5 and penalty_tag_check(t):
                P["messages"] += [f"{tag} is being used very frequently, not inserting in \"{line}\""]
                P["tag"] = None
            else:
                if tag in DEFAULT_PRESETS:
                    P["messages"] += [f"Running custom rules on \"{line}\" for {tag}"]
                    P["line"] = process_tone_rules(line, tag)
                    P["tag"] = None

            for invalid_tag in P['invalid_tags']:
                unknown_tag_log[invalid_tag] += 1

        c = {"line": P['line']}
        if 'tag' in P and P['tag']: c["tag"] = P["tag"]
        if 'messages' in P and P['messages']: c['messages'] = P["messages"]
        cache.append(c)

    save_global_stats(title)
    save_unknown_tag_log(title)

    return cache


def save_global_stats(title):
    df = (pd.DataFrame.from_dict(global_tag_counts, orient="index", columns=["count"])
          .reset_index()
          .rename(columns={"index": "emotion"})
          )
    df = df.sort_values("count", ascending=False)
    file = TAG_STATS_PATH(f'{title}_emotion_stats.csv')
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(file):
        try:
            existing = pd.read_csv(file)
            merged = (
                pd.concat([existing, df])
                .groupby("emotion", as_index=False)
                .sum()
            )
        except Exception:
            merged = df
    else:
        merged = df
    merged = merged.sort_values("count", ascending=False)
    merged.to_csv(file, index=False)


def save_unknown_tag_log(title):
    if not unknown_tag_log:
        return

    df = pd.DataFrame(
        [(tag, count) for tag, count in unknown_tag_log.items()],
        columns=["unknown_emotion", "count"]
    )
    file = TAG_STATS_PATH(f'{title}_emotion_hallucinations.csv')
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(file):
        try:
            existing = pd.read_csv(file)
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
    merged.to_csv(file, index=False)


#------------------------------------------------------------------------------------
#  Logic for post processing after emotion placement.
#  Extract Index from output.
#  Insert Tags to lines
#------------------------------------------------------------------------------------

WINDOW_BACK = 20
WINDOW_FORWARD = 20

with open('Emotions/emotion_dictionary.json') as f:
    DICTIONARY = json.load(f)


def is_negated(sentence, match_start):
    text = sentence.lower()

    # Global idioms
    for idiom in DICTIONARY['NEGATION_IDIOMS']:
        if re.search(idiom, text):
            return True

    # Window around match
    w0 = max(0, match_start - WINDOW_BACK)
    w1 = min(len(text), match_start + WINDOW_FORWARD)
    window = text[w0:w1]

    for neg in DICTIONARY['NEGATION_WORDS']:
        if re.search(neg, window):
            return True

    return False


def char_to_word_index(words, char_pos):
    running = 0
    for idx, w in enumerate(words):
        end = running + len(w)
        if running <= char_pos < end:
            return idx
        running = end + 1
    return None


def lexical_fallback_for_tag(tag, sentence):

    text = sentence.lower()
    patterns = DICTIONARY['SOUND_CUES'].get(tag, {})
    words = sentence.split()
    for pat in patterns:
        m = re.compile(pat).search(text)
        if m:
            start = m.start()
            if not is_negated(sentence, start):
                return char_to_word_index(words, start)

    return None


def extract_index(output):
    try:
        match = re.search(r'\{[^{}]*\}', output)
        if match:
            data = json.loads(match.group(0))
            position = data.get("position")
            return position
    except:
        return ""


def safe_insert(sentence: str, index, tag: str):
    words = sentence.split()
    if index == "END" or index >= len(words):
        return " ".join(words) + f" {tag}"
    return " ".join(words[:index] + [f"{tag}"] + words[index:])


shift_pattern_regex = r'^(.*?)([.!?]["\']?)\s*(<[A-Za-z0-9_]+>)\s*$'


def shift_emotion_inside(sentence):
    # Pattern:
    #   (1) capture everything before final punctuation
    #   (2) capture final punctuation (., ?, !, optionally followed by a quote)
    #   (3) capture the emotion tag at the end: [ANYTHING]
    pattern = re.compile(shift_pattern_regex)

    m = pattern.match(sentence)
    if m:
        before, punctuation, tag = m.groups()
        sentence = f"{before} {tag}{punctuation}"

    return sentence


covert_regex = r'\[([A-Za-z0-9_]+)\]'


def convert_tag(text):
    return re.sub(covert_regex, lambda m: f"<{m.group(1).lower()}>", text)


def process_insertions(L):
    if type(L) == str: return convert_tag(L)

    line = L['line']
    tag = L['tag']
    output = L['pos']
    index = extract_index(output)

    if index is None:
        index = ""
        # print(f"Model couldn't find the index in the output \"{decoded}\"\nDefaulting to END.")

    # Safeguard index
    is_valid_int = isinstance(index, int) and (0 <= index < len(line.split()))
    if not is_valid_int and index != "END":
        index = ""

    if index == "":
        index = lexical_fallback_for_tag(tag[1:-1].lower(), line)
        if not index:
            return line
        index = min(index, len(line.split())-1)

    tag = f"<{tag[1:-1].lower()}>"
    modified_line = safe_insert(line, index, tag)
    modified_line = shift_emotion_inside(modified_line)
    modified_line = re.sub(r"\s+", " ", modified_line).strip()
    return modified_line


################################# M A I N ############################################
def emotion_inst_post_process(lines, title):
    processed_results = [None] * len(lines)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_insertions, l): i for i, l in enumerate(lines)}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"{title}"):
            i = futures[f]
            processed_results[i] = f.result()

    return processed_results
