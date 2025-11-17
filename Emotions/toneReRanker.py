import re

# ---------------------------------------------------------
# BASE STRICT AUDIOBOOK WEIGHTS (unchanged)
# ---------------------------------------------------------

REQUIRED_MIN_SCORE = {
    "angry": 1.0,
    "sarcastic": 1.0,
    "excited": 1.2,
    "curious": 1.4,
    "whisper": 1.3,
    "cry": 1.2,
    "scream": 1.4,
    "sing": 1.4,

    "sigh": 1.2,
    "exhale": 1.3,
    "gasp": 1.0,
    "gulp": 1.4,

    "chuckle": 1.2,
    "giggle": 1.2,
    "laugh": 1.0,
    "laugh_harder": 1.2,
    "snort": 1.4,
}

FALSE_POSITIVE_PENALTY = {
    # Breath / subtle sounds (still risky, but not crippling)
    "sigh": 0.35,
    "exhale": 0.35,
    "gasp": 0.25,

    # Low-intensity social sounds (was too harsh before)
    "curious": 0.30,
    "chuckle": 0.25,
    "giggle": 0.25,
    "whisper": 0.20,

    # Emotional high-confidence tags (keep at zero)
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
    "angry": ["yelled", "shouted", "snapped", "furious", "rage"],
    "sarcastic": ["yeah right", "sure", "oh wow", "really?", "as if"],
    "excited": ["amazing!", "incredible!", "can't wait", "wow!"],
    "curious": ["what's this", "hmm", "tilted his head", "tilted her head"],
    "whisper": ["whispered", "softly", "under her breath", "under his breath"],
    "cry": ["tears", "sobbing", "cried", "choking up"],
    "scream": ["screamed", "shrieked"],
    "sing": ["sang", "singing"],

    "sigh": ["sighed"],
    "exhale": ["exhaled", "let out a breath"],
    "gasp": ["gasped"],
    "gulp": ["gulped", "swallowed hard"],

    "chuckle": ["chuckled"],
    "giggle": ["giggled"],
    "laugh": ["laughed"],
    "laugh_harder": ["laughing harder", "burst out laughing"],
    "snort": ["snorted"],
}

MODERATE_CUES = {
    "angry": ["irritated", "annoyed"],
    "sarcastic": ["smirked"],
    "excited": ["excited", "thrilled"],
    "curious": ["wondered"],

    "chuckle": ["amused"],
    "giggle": ["smiled"],
    "laugh": ["funny"],
}


# ---------------------------------------------------------
# SCORING COMPONENTS
# ---------------------------------------------------------

def score_lexical(tag, text):
    t = text.lower()
    for cue in STRONG_CUES.get(tag, []):
        if cue in t:
            return 2.0
    for cue in MODERATE_CUES.get(tag, []):
        if cue in t:
            return 1.0
    return 0.0


def score_punctuation(text):
    score = 0.0
    if "!" in text:
        score += 0.2
    if "?!" in text:
        score += 0.3
    if "..." in text:
        score += 0.2
    if "—" in text:
        score += 0.2
    return min(score, 0.3)


def score_context(tag, prev_text, next_text):
    prev = prev_text.lower()
    nxt = next_text.lower()

    for cue in STRONG_CUES.get(tag, []):
        if cue in prev or cue in nxt:
            return 0.3

    if tag == "angry" and any(x in nxt for x in ["calmly", "softly"]):
        return -0.5
    if tag == "sigh" and "laughed" in nxt:
        return -0.5

    return 0.0

# ---------------------------------------------------------
# GENRE ADD-ON RULES
# ---------------------------------------------------------


def apply_genre_rules(tag, text, prev_text, next_text, base_score, genre):
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
            base_score += 0.4

        # suppress sigh/exhale unless explicit
        if tag in ["sigh", "exhale"] and "sigh" not in t and "exhal" not in t:
            base_score -= 0.5

        # boost curious for inquisitive questions
        if tag == "curious" and "?" in t:
            base_score += 0.3

        # shy/awkward tension
        if tag in ["chuckle", "giggle"] and any(x in t for x in ["you're ridiculous", "shut up", "don't judge"]):
            base_score += 0.3

        # avoid gasp unless explicit
        if tag == "gasp" and "gasp" not in t:
            base_score -= 0.4

        return base_score

    # -----------------------
    # FANTASY GENRE
    # -----------------------
    if genre == "fantasy":
        # elevate gasp for magical or shocking events
        if tag == "gasp" and any(x in t for x in ["magic", "portal", "ancient", "creature", "power"]):
            base_score += 0.4

        # suppress chuckle/giggle (not common in epic fantasy)
        if tag in ["chuckle", "giggle"]:
            base_score -= 0.4

        # boost whisper for conspiratorial or mystical tones
        if tag == "whisper" and any(x in t for x in ["spell", "ritual", "prophecy", "cloak"]):
            base_score += 0.4

        # epic drama → angry/excited more common
        if tag in ["angry", "excited"] and "!" in t:
            base_score += 0.2

        return base_score

    # -----------------------
    # DRAMA GENRE
    # -----------------------
    if genre == "drama":
        # boost sigh & exhale slightly (drama uses breathiness)
        if tag in ["sigh", "exhale"]:
            base_score += 0.3

        # boost cry if emotional phrasing
        if tag == "cry" and any(x in t for x in ["please", "don't", "why", "can't"]):
            base_score += 0.4

        # reduce laugh-family unless explicit
        if tag in ["chuckle", "giggle", "laugh"]:
            if "laugh" not in t:
                base_score -= 0.4

        return base_score

    return base_score

# ---------------------------------------------------------
# FINAL RERANK FUNCTION WITH GENRE
# ---------------------------------------------------------


def strict_rerank(text, prev_text, next_text, candidate_tags, genre="normal", top_k=2):
    results = []
    score = None
    for tag in candidate_tags:
        les = score_lexical(tag, text)
        pps = score_punctuation(text)
        ccs = score_context(tag, prev_text, next_text)
        fpp = FALSE_POSITIVE_PENALTY[tag]
        required = REQUIRED_MIN_SCORE[tag]

        base_score = (les + pps + ccs) - fpp

        # apply genre rules
        final_score = apply_genre_rules(tag, text, prev_text, next_text, base_score, genre)

        score = final_score
        if final_score >= required:
            results.append((tag, final_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [tag.upper() for tag, _ in results[:top_k]], round(score, 2)

# ---------------------------------------------------------
# Example
# ---------------------------------------------------------


if __name__ == "__main__":
    prev_s = "He stared into the flickering torchlight."
    curr_s = "She whispered the old prophecy under her breath."
    next_s = "The cavern trembled softly."

    model_tags = ["EXCITED", "CURIOUS", "WHISPER"]

    print(strict_rerank(curr_s, prev_s, next_s, [t.lower() for t in model_tags], genre="fantasy"))