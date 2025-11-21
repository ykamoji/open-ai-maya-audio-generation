import re
import math

# ---------------------------------------------------------
# BASE AUDIOBOOK WEIGHTS
# ---------------------------------------------------------

REQUIRED_MIN_SCORE = {
    "angry": 0.65,
    "sarcastic": 0.60,
    "excited": 0.60,
    "curious": 0.60,
    "whisper": 0.60,
    "cry": 0.60,
    "scream": 0.65,
    "sing": 0.65,

    "appalled": 0.60,
    "mischievous": 0.55,
    "disappointed": 0.55,

    "sigh": 0.55,
    "exhale": 0.55,
    "gasp": 0.55,
    "gulp": 0.65,

    "chuckle": 0.55,
    "giggle": 0.55,
    "laugh": 0.60,
    "laugh_harder": 0.65,
    "snort": 0.70
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
# LEXICAL CUES
# ---------------------------------------------------------

STRONG_CUES = {
    "angry": [
        "yell", "shout", "yelled", "shouted", "snap", "furious", "rage", "angrily", "mad", "irritated", "annoyed",
        "fuming", "gritted his teeth", "snarl", "glared", "eyes burned", "stomped", "slammed", "shook with anger"
    ],
    "sarcastic": [
        "yeah right", "sure", "oh wow", "really?", "as if", "right...", "mocking", "ironic", "snarky",
        "dry tone", "slow clap", "mockingly", "dry tone", "smirked", "said with a smirk", "said flatly",
        "deadpan"
    ],
    "excited": [
        "amazing!", "incredible!", "can't wait", "wow!", "so excited", "thrill", "eager", "enthusiast",
        "pumped", "lively", "eyes lit up", "leaned forward eagerly"
    ],
    "curious": [
        "what's this", "hmm", "tilted his head", "tilted her head", "wonder", "inquisitive", "question",
        "puzzle", "brows lifted", "eyes narrowed in thought", "studied him", "peered", "examined", "questioned"
    ],
    "whisper": [
        "whispered", "softly", "murmur", "hush", "soft voice", "quiet voice", "under her breath", "under her breath",
        "barely audible", "hushed voice"
    ],
    "cry": [
        "tears", "cried", "choking up", "crying", "sobs", "weep", "whimper", "bawl",
        "tears", "sobbing", "choking up", "voice broke", "weeping", "sniffling", "tear-streaked", "sobbed"
    ],
    "scream": [
        "yell", "shout", "shriek", "screamed", "shrieked", "yelled!", "shouted!", "howled",
        "voice cracked from shouting", "let out a scream"
    ],
    "sing": [
        "sang", "singing", "melody", "humming", "tune", "softly singing"
    ],
    "sigh": [
        "sigh", "sighed", "weary", "exasperated", "breath out", "long breath", "long sigh", "exasperated",
        "let out a sigh", "breath escaped", "shoulders slumped"
    ],
    "exhale": [
        "exhaled", "let out a breath", "released a breath", "breath rushed out", "long breath"
    ],
    "gasp": [
        "gasp", "gasped", "sharp breath", "eyes widened", "whoa", "inhale sharply", "breath hitched",
        "intake of breath", "swallow hard", "tight throat", "startled"
    ],
    "gulp": [
        "gulped", "swallowed hard", "tight throat", "dry swallow"
    ],
    "chuckle": [
        "laughs softly", "chuckled", "small laugh", "amused sound", "low laugh"
    ],
    "giggle": [
        "giggled"
    ],
    "laugh": [
        "laughed", "funny", "burst into laughter", "couldn't stop laughing", "cracked up", "laughed out loud"
    ],
    "laugh_harder": [
        "burst out laughing", "laughing harder", "doubling over in laughter",
        "roared with laughter" "couldn't hold back laughter"
    ],
    "snort": [
        "snicker",
    ],
    "appalled": [
        "appalled", "disgusted", "horrified", "recoiled",
        "face twisted in disgust", "staggered back", "what on earth"
    ],
    "mischievous": [
        "grin", "glint in his eye", "glint in her eye",
        "smirked", "playful smirk", "grinned slyly", "scheming"
    ],
    "disappointed": [
        "let down", "crestfallen", "slumped", "face fell", "voice fell", "eyes dimmed", "deep sigh"
    ],
}

MODERATE_CUES = {
    "angry": [
         "annoyed", "irritated", "tense voice", "short tone", "frustrated", "brows furrowed", "sharp tone"
    ],
    "sarcastic": [
        "smirked", "dryly", "flat tone"
    ],
    "excited": [
        "thrilled", "eager", "anticipation", "buzzing", "louder", "rising energy", "excitement",
        "leaned forward", "eyes sparkled", "loud"
    ],
    "curious": [
        "curious", "wondering", "leaned in slightly", "thoughtful tone", "studying", "intrigued"
    ],
    "whisper": [
        "quietly", "low voice", "soft tone", "barely said"
    ],
    "cry": [
        "voice wavered", "eyes watered", "emotional", "on the verge of tears"
    ],
    "scream": [
        "raised voice", "shouting tone", "high-pitched voice"
    ],
    "sing": [
        "light humming", "soft tune", "melodic voice"
    ],

    "sigh": [
        "tired tone", "heavy breath", "felt drained"
    ],
    "exhale": [
        "breath escaped", "slow breath", "steadying breath"
    ],
    "gasp": [
        "breath caught", "sharp inhale", "surprised"
    ],
    "gulp": [
        "nervous swallow", "dry mouth"
    ],
    "chuckle": [
        "amused", "smiled lightly"
    ],
    "giggle": [
        "smiled", "light laugh"
    ],

    "laugh": [
        "funny", "soft laugh", "amused tone"
    ],
    "laugh_harder": [
        "laughing more", "couldn't hold back laughter"
    ],
    "snort": [
        "amused breath"
    ],
    "appalled": [
        "pulled back slightly", "stepped away", "brows lifted sharply"
    ],
    "mischievous": [
        "smirked lightly", "eyes sparkled with intent",
        "playful look", "tilted head with a grin"
    ],
    "disappointed": [
        "looked down", "quiet tone", "shoulders lowered", "tone softened", "weak smile"
    ],
}

WEAK_CUES = {
    "excited": [
        "energy", "buzzing", "rumble", "roar", "steps quickened", "anticipation built",
        "air vibrated", "waves of sound", "movement all around",
        "lights brightened", "hustle of people", "energy rising",
        "felt alive", "heart picked up", "quickened pace",
        "could feel it in the air", "electric atmosphere"
        "quick movements", "brisk pace", "bright surroundings",
        "momentum grew", "felt the pull forward", "alive with motion", "cheers"
    ],

    "curious": [
        "studied", "peered", "tilted", "leaned in", "observed",
        "examined", "looked closer", "glanced around", "lingered gaze",
        "scanned the area", "brows raised slightly", "head tilted",
        "traced the lines", "paused to look", "eyes followed",
        "inspect", "unknown sight", "caught attention",
        "shifted focus", "drawn toward"
    ],

    "sigh": [
        "long moment", "heavy silence", "quiet settled",
        "weight of the", "took a moment", "rested her shoulders",
        "rested his shoulders", "slowed to a stop", "stillness hung",
        "lingering quiet", "moment stretched", "breath softened",
        "sank slightly", "lowered posture", "quiet pause",
        "time seemed to slow", "let things settle"
    ],

    "exhale": [
        "breath escaped", "release of air", "tension leaving",
        "slow air", "steadying himself", "steadying herself",
        "loosening grip", "relaxing slightly", "chest lowered",
        "air drifted", "in the stillness", "dropped his shoulders",
        "dropped her shoulders", "breath slipped out"
    ],

    "gasp": [
        "froze", "went still", "eyes wide", "eyes widened",
        "heart skipped", "shock ran through", "stopped short",
        "pulled back slightly", "stiffened", "halted mid-step",
        "sudden silence", "everything stopped", "sharp movement",
        "jerked back", "stumbled for a moment"
    ],

    "gulp": [
        "swallowed", "tight throat", "dry throat", "nervous stillness",
        "rigid posture", "shifted nervously", "unsteady hands",
        "backed up a step", "looked uneasy", "hesitant pause",
        "mouth tightened", "lips pressed together", "averted eyes",
        "leaned away slightly"
    ],

    "angry": [
        "jaw tightened", "tension rose", "sharp look", "muscles tensed",
        "brows narrowed", "steps heavy", "slammed down his foot",
        "shoulders locked", "hands tightened", "eyes hardened",
        "tone dropped", "movement stiff", "cold stare",
        "clenched posture", "held firm", "unblinking glare"
    ],

    "whisper": [
        "leaned close", "brought face closer", "steps softened",
        "moved quietly", "softened posture", "ducked head slightly",
        "held breath for a moment", "lowered voice area",
        "dimly lit space", "shadows stretched long"
    ],

    "cry": [
        "voice thin", "words faltered", "breathing uneven",
        "looked down at the ground", "held her arm tightly",
        "held his arm tightly", "wavered", "trembled slightly",
        "blinked a few times", "blinked rapidly", "voice nearly broke",
        "edges of voice softened", "stillness of the moment"
    ],

    "scream": [
        "rising panic", "frantic movement", "backed away quickly",
        "scrambled", "chaos erupted",
        "crack of tension",
        "surge of adrenaline", "disturbance ahead"
    ],

    "laugh": [
        "loosening tension", "face brightened", "smile forming",
        "a bit lighter", "moment eased", "soft warmth",
        "playful energy", "air shifted brighter"
    ],

    "chuckle": [
        "small smile", "lips curled slightly", "eyes softened",
        "tiny smirk", "easy movement", "relaxed a little"
    ],

    "giggle": [
        "lightness in the air", "small bounce in her step",
        "tiny grin", "playful eyes"
    ],

    "laugh_harder": [
        "bending forward slightly", "clutching stomach lightly",
        "face lighting up strongly"
    ],

    "snort": [
        "short breath out", "huffed lightly", "quick puff of air"
    ],

    "sing": [
        "rhythmic movement", "soft humming rhythm",
        "gentle cadence", "flowing tone in the air"
    ],
    "appalled": [
        "froze in place", "stopped mid-step", "expression shifted", "leaned back slightly", "hesitated visibly",
        "pulled her hand back", "pulled his hand back", "drew back a little", "took a half step back",
        "went quiet suddenly", "room fell still", "eyes darted away", "stiffened for a moment",
        "blinked twice", "blinked rapidly", "mouth opened slightly", "jaw slackened for a moment",
        "air seemed to thin", "movement halted briefly", "went rigid", "grip loosened slightly",
        "voice caught for a second", "breath faltered", "tension rippled through",
    ],
    "mischievous": [
        "hint of a grin", "tiny grin forming", "glanced sideways", "glance to the side",
        "eyes narrowed playfully", "eyes sparkled briefly", "small smirk", "corner of mouth lifted",
        "shifted weight lightly", "rocked on heels", "tilted her head with interest", "tilted his head with interest",
        "hands clasped behind back", "hands tucked behind", "stepped closer with ease", "leaned in slightly",
        "felt the moment lighten", "air grew lighter", "whisper of amusement", "playful glint",
        "tried to hide a smile", "nearly broke into a smile", "pretended not to notice", "looked far too casual",
    ],
    "disappointed": [
        "quiet for a moment", "voice low", "moment stretched out", "air grew still",
        "looked down briefly", "glanced at the floor", "slowed his step", "slowed her step",
        "shoulders dipped slightly", "posture loosened", "let the silence linger", "fell quiet again",
        "breath softened", "voice softened", "held her arm lightly", "held his arm lightly",
        "stared at the ground", "stared past him", "lowered his gaze", "lowered her gaze",
        "hands relaxed at sides", "hands dropped slightly",
        "exhaled without noticing", "paused mid-sentence",
        "tone faded briefly", "words trailed off", "but I couldn't"
    ],
}


# ---------------------------------------------------------
# SCORING LEXICAL
# ---------------------------------------------------------


def  score_lexical(tag, text):
    t = text.lower()
    tag = tag.lower()

    score = 0.0

    # --------------------------------------------------------
    # 1. Morphological match: laugh/laughs/laughed/laughing
    # --------------------------------------------------------
    morph_tag = r"\b" + re.escape(tag) + r"(s|ed|ing)?\b"
    if re.search(morph_tag, t):
        score += 2.0

    # --------------------------------------------------------
    # 2. STRONG CUES (priority)
    # --------------------------------------------------------
    for cue in STRONG_CUES.get(tag, []):
        cue_l = cue.lower()
        if re.search(rf"\b{re.escape(cue_l)}\b", t):
            score += 2.0

    # --------------------------------------------------------
    # 3. MODERATE CUES
    # --------------------------------------------------------
    for cue in MODERATE_CUES.get(tag, []):
        cue_l = cue.lower()
        if re.search(rf"\b{re.escape(cue_l)}\b", t):
            score += 1.0

    # --------------------------------------------------------
    # 4. Week CUES
    # --------------------------------------------------------
    for cue in WEAK_CUES.get(tag, []):
        cue_l = cue.lower()
        if re.search(rf"\b{re.escape(cue_l)}\b", t):
            score += 0.5

    # --------------------------------------------------------
    # 5. NO MATCH
    # --------------------------------------------------------
    return score


def lexical_candidate_tags(text) :
    candidates = []
    for tag in REQUIRED_MIN_SCORE.keys():  # all known tags
        raw_lex = score_lexical(tag, text)
        if raw_lex >= 1.5:
            candidates.append(tag)
    return candidates

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


def rerank(text, candidate_tags, genre="normal", top_k=2):
    results = []
    LLM_prob = [0.95, 0.45, 0.1]
    lex_candidates = lexical_candidate_tags(text)

    for lex_cand in lex_candidates:
        if lex_cand not in candidate_tags:
            candidate_tags.append(lex_cand)

    for i, tag in enumerate(candidate_tags):
        LLM_score = LLM_prob[i]
        les = score_lexical(tag, text)
        if les < 1.5:
            les = 0.0
        fpp = FALSE_POSITIVE_PENALTY[tag]
        required = REQUIRED_MIN_SCORE[tag]

        base_score = 0.6 * LLM_score + 0.4 * les - fpp

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
    curr_s = "That made me feel like he knows just how strong this person is."

    model_tags = ['CURIOUS']

    print(rerank(curr_s, [t.lower() for t in model_tags], genre="YA"))
