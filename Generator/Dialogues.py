import re


def dialogue_split(paragraph):
    dialogues = paragraph.split("\"")
    lines = []
    if len(dialogues) == 1:
        return paragraph, False

    counter = 0
    for line in dialogues:
        counter += 1
        if line.strip() == "":
            counter -= 1
        elif counter % 2 == 0:
            # Quoted line
            lines.append("\"" + line.strip() + "\"")
        else:
            # Normal line
            lines.append(line.strip())

    return lines, True


punctuations = [',', '!', '?', '.', ':', '"']


def combine_lines(sentences, limit):
    lines = []
    current = ""
    for m in sentences:
        if len(current.split()) + len(m.split()) <= limit:
            current += m.strip() + (" " if m.strip()[-1] in punctuations else ". ")
        else:
            if current.strip() != "":
                lines.append(current.strip())
            current = m.strip() + (" " if m.strip()[-1] in punctuations else ". ")
    if current.strip() != "":
        lines.append(current.strip() + ("" if current.strip()[-1] in punctuations else "."))
    return lines


INIT_ABBR_RE = re.compile(r"\b(?:[A-Za-z]\.){1,5}")
TITLE_ABBR_RE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.")
DOT = "∯"


def protect_abbreviations(text: str) -> str:
    text = INIT_ABBR_RE.sub(lambda m: m.group().replace(".", DOT), text)
    text = TITLE_ABBR_RE.sub(lambda m: m.group().replace(".", DOT), text)
    return text


def split_language_sentences(sentence):
    protected = protect_abbreviations(sentence)
    parts = re.split(r"\.\s*", protected)
    monologues = [p.replace(DOT, ".").strip() for p in parts if p.strip()]
    return monologues


def align_monologue(paragraph, limit):
    monologues = split_language_sentences(paragraph)
    lines = combine_lines(monologues, limit)
    return lines


def clean_dialogue(dialogues, limit):

    lines = []
    current = ""
    try:
        for d in dialogues:
            if (not current.endswith(".")) and d.startswith("\""):
                if current.endswith(",") or current.endswith("!") or current.endswith("?") or current.endswith(":"):
                    current += " " + d
                else:
                    current += ". " + d
            elif current.endswith("\"") and (bool(d) and d[0].isalpha() and d[0].islower()):
                current += " " + d
            else:
                if current.strip() != "":
                    lines += [current + ("" if current[-1] in punctuations else ".")]
                current = d

        if current.strip() != "":
            lines += [current + ("" if current[-1] in punctuations else ".")]

    except Exception as e:
        print(f"Error in dialogue reconstruction {e} for {current}")

    combined = []
    current = ""
    try:
        for l in lines:
            if len(current.split()) + len(l.split()) <= limit:
                current += " " + l.strip()
            else:
                if current.strip() != "":
                    combined.append(current.strip())
                current = l.strip()

        if current.strip() != "":
            combined.append(current.strip())
    except Exception as e:
        print(f"Error in dialogue combining {e} for {current}")

    return combined


def align_dialogue(dialogues):
    lines = []
    for d in dialogues:
        try:
            if '"' in d:
                lines += [d]
            else:
                monologues = split_language_sentences(d)
                n = len(monologues)
                for i in range(n):
                    if i < n - 1:
                        next_word = monologues[i+1]
                        if bool(next_word) and next_word[0].isalpha() and next_word[0].isupper():
                            lines += [monologues[i] + ("" if monologues[i][-1] in punctuations else ".")]
                        else:
                            lines += [monologues[i]]
                    else:
                        lines += [monologues[i]]

        except Exception as e:
            print(f"Error {e} for aligning dialogue {d}")

    return lines


def split_expressions(line):
    ## Since there is only one tag per sentence, multiple tags means multiple sentences,
    # split based the first punctuation: [! ? or . ].
    ## Since this is after dialogue processing, need to check for balanced quotes so that middle sentences are not cut.

    n = len(line)
    i = 0
    j = 0
    single_expressions = []
    quote_balanced = 0
    try:
        while i < n:
            if line[i] == '"':
                quote_balanced += 1
            elif line[i] == '.' or line[i] == '!' or line[i] == '?':

                if quote_balanced % 2 == 0:
                    single_expressions.append(line[j:i+1].strip())
                    j = i + 1
            i += 1

        if j < i:
            single_expressions.append(line[j:].strip())
    except Exception as e:
        print(f"Error {e} for splitting expressions for {line}")

    return single_expressions

TAG_CHECK = re.compile(r"<[^<>]+>")


def expression_aware_splitting(lines):

    expression_lines = []
    for l in lines:
        expressions_count = TAG_CHECK.findall(l)
        if len(expressions_count) > 1:
            expression_lines.extend(split_expressions(l))
        else:
            expression_lines.append(l)

    return expression_lines


def process(paragraphs, limit):
    chunks = []
    para_breaks = []
    broken_paragraphs = []
    for para in paragraphs:

        # If paragraphs has unbalanced quotes, stop the execution. Fix it manually.
        if para.count('"') % 2 != 0:
            broken_paragraphs.append(para)
            continue

        # Stage 1:
        # If paragraph is just monologue, return it.
        # If a paragraph has quotes, return quote-aware lines.
        stage1, complexity = dialogue_split(para)

        if not complexity:
            # When the paragraph is just a monologue, split it by normal sentence splitting logic with limits.
            para_lines = align_monologue(stage1, limit)

        else:
            # Stage 2: Align the dialogues by first forming the monologues in between the dialog's
            stage2 = align_dialogue(stage1)

            # Stage 2:
            # Clean the remaining dialogues
            # Apply the limits
            para_lines = clean_dialogue(stage2, limit)

        # After all dialogue processing, need to split sentences having multiple expressions to single.
        # This logic removes around 95 % of the issues. Remaining cases are complex and should be manually fixed.
        expression_lines = expression_aware_splitting(para_lines)

        # For some reason, the model puts out double spaces instead of commas. Fixing it here, at the final dialogues.
        expression_lines = [exp.replace("  ", ", ") for exp in expression_lines]

        chunks.extend(expression_lines)

        para_breaks.append(len(chunks))

    tagged_list = [
        bool(TAG_CHECK.search(c))
        for c in chunks
    ]

    return chunks, tagged_list, para_breaks, broken_paragraphs
