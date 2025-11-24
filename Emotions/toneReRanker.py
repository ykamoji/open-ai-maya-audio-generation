import re
import json

with open('Emotions/emotion_dictionary.json') as f:
    CUE_DICTIONARY = json.load(f)

# ---------------------------------------------------------
# BASE AUDIOBOOK WEIGHTS
# ---------------------------------------------------------


REQUIRED_MIN_SCORE = {
    "angry": 0.55,
    "sarcastic": 0.50,
    "excited": 0.50,
    "curious": 0.50,
    "whisper": 0.50,
    "cry": 0.50,
    "scream": 0.55,
    "sing": 0.50,

    "appalled": 0.50,
    "mischievous": 0.45,
    "disappointed": 0.45,

    "sigh": 0.45,
    "exhale": 0.45,
    "gasp": 0.45,
    "gulp": 0.55,

    "chuckle": 0.45,
    "giggle": 0.45,
    "laugh": 0.50,
    "laugh_harder": 0.55,
    "snort": 0.60
}

FALSE_POSITIVE_PENALTY = {
    "sigh": 0.12,
    "exhale": 0.10,
    "gasp": 0.08,

    "appalled": 0.10,
    "mischievous": 0.00,
    "disappointed": 0.08,

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
# SCORING LEXICAL
# ---------------------------------------------------------

def morphs(tag):
    return  r"\b" + re.escape(tag) + r"(s|ed|ing|ly|er)?\b"


def score_lexical(tag, text):
    t = text.lower()
    tag = tag.lower()

    score = 0.0

    # --------------------------------------------------------
    # 1. Morphological match: laugh/laughs/laughed/laughing
    # --------------------------------------------------------

    if re.search(morphs(tag), t):
        score += 0.3

    # --------------------------------------------------------
    # 2. STRONG CUES (priority)
    # --------------------------------------------------------
    for cue in CUE_DICTIONARY['STRONG_CUES'].get(tag, []):
        cue_l = cue.lower()
        if re.search(morphs(cue_l), t):
            score += 0.25

    # --------------------------------------------------------
    # 3. MODERATE CUES
    # --------------------------------------------------------
    for cue in CUE_DICTIONARY['MODERATE_CUES'].get(tag, []):
        cue_l = cue.lower()
        if re.search(cue_l, t):
            score += 0.15

    # --------------------------------------------------------
    # 4. Week CUES
    # --------------------------------------------------------

    semantic_score = 0
    families = CUE_DICTIONARY['SEMANTIC_CUE_DICTIONARY'].get(tag, [])
    for fam in families:
        patterns = CUE_DICTIONARY['MACRO_PATTERNS'].get(fam, [])
        for p in patterns:
            if semantic_score >= 0.6:
                break
            if re.search(p, t):
                semantic_score += 0.2

    score += semantic_score

    weak_score = 0
    for cue in CUE_DICTIONARY['WEAK_PATTERNS'].get(tag, []):
        if weak_score >= 0.40:
            break
        cue_l = cue.lower()
        if re.search(cue_l, t):
            weak_score += 0.08

    score += weak_score
    # --------------------------------------------------------
    # 5. NO MATCH
    # --------------------------------------------------------
    return score


def lexical_candidate_tags(text) :
    candidates = []
    for tag in REQUIRED_MIN_SCORE.keys():  # all known tags
        raw_lex = score_lexical(tag, text)
        if raw_lex >= 0.18:
            candidates.append(tag)
    return candidates


def custom_negative_patterns(tag, text):

    score = 0
    negatives = CUE_DICTIONARY['NEGATIVE_PATTERNS'].get(tag, [])
    for neg in negatives:
        if score >= 0.5:
            break
        if re.search(neg, text):
            score += 0.25

    return score

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
            base_score += 0.15

        # shy/awkward tension
        if tag in ["chuckle", "giggle"] and any(x in t for x in ["you're ridiculous", "shut up", "don't judge"]):
            base_score += 0.06

        # avoid gasp unless explicit
        if tag == "gasp" and "gasp" not in t:
            base_score -= 0.06

        if tag == "mischievous":
            base_score += 0.04

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

LLM_prob = [0.95, 0.55]


def rerank(text, model_tags, genre="normal", top_k=2):
    results = []

    candidate_tags = list(model_tags)

    for cand in lexical_candidate_tags(text):
        if cand not in candidate_tags:
            candidate_tags.append(cand)

    for i, tag in enumerate(candidate_tags):

        lex_score = score_lexical(tag, text)
        if tag in model_tags:
            LLM_score = LLM_prob[i]
        else:
            LLM_score = 0.30 + 0.40 * min(1.0, lex_score / 0.85)

        fpp = FALSE_POSITIVE_PENALTY[tag]
        cfpp = custom_negative_patterns(tag, text)
        required = REQUIRED_MIN_SCORE[tag]

        base_score = 0.35 * LLM_score + 0.65 * lex_score - fpp - cfpp

        # apply genre rules
        final_score = round(apply_genre_rules(tag, text, base_score, genre), 2)

        if final_score >= required:
            results.append((tag, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:top_k]
    return [f"[{tag.upper()}]" for tag, _ in results], [f"{round(score, 2)}/{REQUIRED_MIN_SCORE[tag]}" for tag, score in
                                                        results]


# ---------------------------------------------------------
# Example
# ---------------------------------------------------------

if __name__ == "__main__":
    curr_s = "As soon as I heard him walk towards him in shock, I faced him, my voice rising."

    model_tags = ['ANGRY','EXCITED']

    print(rerank(curr_s, [t.lower() for t in model_tags], genre="YA"))
