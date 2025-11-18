import os
import re
import json
# ---------------------------------------------------------
# BASE AUDIOBOOK WEIGHTS
# ---------------------------------------------------------

REQUIRED_MIN_SCORE = {
    "angry": 0.55,
    "sarcastic": 0.55,
    "excited": 0.60,
    "curious": 0.60,
    "whisper": 0.60,
    "cry": 0.60,
    "scream": 0.65,
    "sing": 0.65,

    "sigh": 0.55,
    "exhale": 0.55,
    "gasp": 0.55,
    "gulp": 0.65,

    "chuckle": 0.55,
    "giggle": 0.55,
    "laugh": 0.50,
    "laugh_harder": 0.65,
    "snort": 0.70,
}

FALSE_POSITIVE_PENALTY = {
    "sigh": 0.12,
    "exhale": 0.10,
    "gasp": 0.08,

    "curious": 0.10,
    "chuckle": 0.08,
    "giggle": 0.08,
    "whisper": 0.10,

    "angry": 0.0,
    "sarcastic": 0.0,
    "excited": 0.0,
    "cry": 0.0,
    "scream": 0.0,
    "sing": 0.0,
    "laugh": 0.0,
    "laugh_harder": 0.0,
    "snort": 0.0,
    "gulp": 0.0,
}

# ---------------------------------------------------------
# LEXICAL CUES
# ---------------------------------------------------------

STRONG_CUES = {
    "angry": ["yell", "shout", "snap", "furious", "rage", "angrily", "mad", "irritated", "annoyed", "fuming"],
    "sarcastic": ["yeah right", "sure", "oh wow", "really?", "as if", "right...", "mocking", "ironic", "snarky",
                  "dry tone"],
    "excited": ["amazing!", "incredible!", "can't wait", "wow!", "so excited", "thrill", "eager", "enthusiast",
                "pumped", "lively"],
    "curious": ["what's this", "hmm", "tilted his head", "tilted her head", "wonder", "inquisitive", "question",
                "puzzle"],
    "whisper": ["whispered", "softly", "murmur", "hush", "soft voice", "quiet voice", "under her breath"],
    "cry": ["tear", "cried", "choking up", "crying", "sob", "weep", "whimper", "bawl"],
    "scream": ["yell", "shout", "shriek"],
    "sing": ["sang", "singing", "melody", "humming"],

    "sigh": ["sigh", "sighed", "weary", "exasperated", "breath out", "long breath"],
    "exhale": ["exhaled", "let out a breath", "long breath", "breathe out", "breath release"],
    "gasp": ["gasp", "gasped", "sharp breath", "eyes widened", "whoa", "inhale sharply", "breath hitched",
             "intake of breath", "swallow hard", "tight throat"],
    "gulp": ["gulped", "swallowed hard"],

    "chuckle": ["laugh softly"],
    "giggle": ["giggled"],
    "laugh": ["laughed", "funny","snicker", "snort"],
    "laugh_harder": ["burst out laughing", "laughing harder"],
    "snort": ["snicker"],
}

MODERATE_CUES = {
    "angry": ["irritated", "annoyed"],
    "sarcastic": ["smirked"],
    "excited": ["excited", "thrilled"],
    "curious": ["wondering", "curious"],
    "chuckle": ["amused"],
    "giggle": ["smiled"],
    "laugh": ["funny"],
}


def build_synonym_map():
    syn_map = {}

    # Strong cues first
    for tag, cues in STRONG_CUES.items():
        syn_map.setdefault(tag, [])
        syn_map[tag].extend(cues)

    # Moderate cues next
    for tag, cues in MODERATE_CUES.items():
        syn_map.setdefault(tag, [])
        syn_map[tag].extend(cues)

    return syn_map


SYNONYM_MAP = build_synonym_map()


# ---------------------------------------------------------
# SCORING LEXICAL
# ---------------------------------------------------------


def score_lexical(tag, text, generated_cues=[]):
    if generated_cues is None:
        generated_cues = []
    t = text.lower()
    tag = tag.lower()

    # --------------------------------------------------------
    # 1. Morphological match: laugh/laughs/laughed/laughing
    # --------------------------------------------------------
    morph_tag = r"\b" + re.escape(tag) + r"(s|ed|ing)?\b"
    if re.search(morph_tag, t):
        return 1.0

    # --------------------------------------------------------
    # 2. STRONG CUES (priority)
    # --------------------------------------------------------
    for cue in STRONG_CUES.get(tag, []):
        cue_l = cue.lower()

        # direct phrase match
        if cue_l in t:
            return 2.0

        # morph match on first word
        first = cue_l.split()[0]
        morph = r"\b" + re.escape(first) + r"(s|ed|ing)?\b"
        if re.search(morph, t):
            return 2.0

    # --------------------------------------------------------
    # 3. MODERATE CUES
    # --------------------------------------------------------
    for cue in MODERATE_CUES.get(tag, []):
        cue_l = cue.lower()

        if cue_l in t:
            return 1.0

        first = cue_l.split()[0]
        morph = r"\b" + re.escape(first) + r"(s|ed|ing)?\b"
        if re.search(morph, t):
            return 1.0

    # --------------------------------------------------------
    # 4. AUTO SYNONYMS (from cue dictionaries)
    # --------------------------------------------------------
    for syn in SYNONYM_MAP.get(tag, []):
        syn_l = syn.lower()

        # direct check
        if syn_l in t:
            return 1.0

        # morph check for synonyms
        first = syn_l.split()[0]
        morph = r"\b" + re.escape(first) + r"(s|ed|ing)?\b"
        if re.search(morph, t):
            return 1.0

    # --------------------------------------------------------
    # 5. GENERATED CUES (JSON custom learned cues)
    # --------------------------------------------------------
    for entry in generated_cues.get(tag, []):
        if entry["cue"].lower() in t:
            return entry["weight"]
    for cues in generated_cues.get(tag, []):
        if cues['cue'] in t:
            return cues['weight']

    # --------------------------------------------------------
    # 6. NO MATCH
    # --------------------------------------------------------
    return 0.0

# ---------------------------------------------------------
# GENRE ADD-ON RULES
# ---------------------------------------------------------


def apply_genre_rules(tag, text, base_score, genre):
    t = text.lower()

    # -----------------------
    # NORMAL (strict default)
    # -----------------------
    if genre == "normal":
        return base_score

    # -----------------------
    # YA GENRE
    # -----------------------
    if genre == "YA":
        # sarcasm boost for banter
        if tag == "sarcastic" and any(x in t for x in ["right.", "sure.", "wow", "okay"]):
            base_score += 0.05

        # suppress sigh/exhale unless explicit
        if tag in ["sigh", "exhale"] and "sigh" not in t and "exhale" not in t:
            base_score -= 0.06

        # boost curious for inquisitive questions
        if tag == "curious" and "?" in t:
            base_score += 0.05

        # shy/awkward tension
        if tag in ["chuckle", "giggle"] and any(x in t for x in ["you're ridiculous", "shut up", "don't judge"]):
            base_score += 0.06

        # avoid gasp unless explicit
        if tag == "gasp" and "gasp" not in t:
            base_score -= 0.06

        return base_score

    # -----------------------
    # FANTASY GENRE
    # -----------------------
    if genre == "fantasy":
        # elevate gasp for magical or shocking events
        if tag == "gasp" and any(x in t for x in ["magic", "portal", "ancient", "creature", "power"]):
            base_score += 0.06

        # suppress chuckle/giggle (not common in epic fantasy)
        if tag in ["chuckle", "giggle"]:
            base_score -= 0.05

        # boost whisper for conspiratorial or mystical tones
        if tag == "whisper" and any(x in t for x in ["spell", "ritual", "prophecy", "cloak"]):
            base_score += 0.06

        # epic drama → angry/excited more common
        if tag in ["angry", "excited"] and "!" in t:
            base_score += 0.05

        # boost curious for inquisitive questions
        if tag == "curious" and "?" in t:
            base_score += 0.04

        return base_score

    # -----------------------
    # DRAMA GENRE
    # -----------------------
    if genre == "drama":
        # boost sigh & exhale slightly (drama uses breathiness)
        if tag in ["sigh", "exhale"]:
            base_score += 0.06

        # boost cry if emotional phrasing
        if tag == "cry" and any(x in t for x in ["please", "don't", "why", "can't"]):
            base_score += 0.08

        # reduce laugh-family unless explicit
        if tag in ["chuckle", "giggle", "laugh"] and "laugh" not in t:
            base_score -= 0.06

        return base_score

    return base_score

# ---------------------------------------------------------
# FINAL RERANK FUNCTION WITH GENRE
# ---------------------------------------------------------


def rerank(text, candidate_tags, genre="normal", top_k=2):

    generated_cues = []
    if os.path.isfile('emotion_cue_weights.json'):
        with open('emotion_cue_weights.json') as f:
            generated_cues = json.load(f)

    results = []
    LLM_prob = [0.7, 0.3, 0.2]
    for i, tag in enumerate(candidate_tags):
        LLM_score = LLM_prob[i]
        les = score_lexical(tag, text, generated_cues)
        fpp = FALSE_POSITIVE_PENALTY[tag]
        required = REQUIRED_MIN_SCORE[tag]

        base_score = 0.7 * LLM_score + 0.3 * les - fpp

        # apply genre rules
        final_score = apply_genre_rules(tag, text, base_score, genre)

        if final_score >= required:
            results.append((tag, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:top_k]
    return [f"[{tag.upper()}]" for tag, _ in results], [f"{round(score,2)}/{REQUIRED_MIN_SCORE[tag]}" for tag, score in results]


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":
    curr_s = "I usually just kick the main door, you know.\" He laughs."

    model_tags = ["LAUGH"]

    print(rerank(curr_s, [t.lower() for t in model_tags], genre="fantasy"))