import re

pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.)\s'


def createChunks(content, limit=None):

    chunks = []
    paragraphs = [p.strip("\n").strip() for p in content.split("\n\n") if p.strip()]
    if limit is not None:
        chunks = paraChunks(paragraphs, limit)
    else:
        for para in paragraphs:
            lines = convert_to_sentences(para)
            chunks.extend(lines)
            chunks.append('')

    return chunks


def convert_to_sentences(content):
    return [se for se in re.split(pattern, content) if se.strip()]


def batch_sentences(lines, limit=300):
    result = [lines[0], ""]
    current = ""
    for line in lines[1:]:
        if line.strip() == "":
            if current:
                result.append(current.strip())
                current = ""
            result.append("")
            continue

        if len(current) + len(line) + (1 if current else 0) > limit:
            if current:
                result.append(current.strip())
            current = line
        else:
            current = (current + " " + line).strip() if current else line

    if current:
        result.append(current.strip())

    return result


def paraChunks(paragraphs, limit):
    chunks = []
    for para in paragraphs:
        if len(para) >= limit:
            lines = [line for line in re.split(pattern, para) if line.strip()]
            counter = 0
            i = 0
            split_pos = [0]
            while i < len(lines):
                counter += len(lines[i])
                if counter >= limit:
                    split_pos.append(i - 1)
                    counter = len(lines[i])
                i += 1

            split_len = len(split_pos)
            for s in range(1, split_len):
                begin = split_pos[s - 1]
                end = min(split_pos[s], split_len)
                chunks.append(". ".join(lines[begin:end]))
        else:
            chunks.append(para)

    # Safe check
    for index, chunk in enumerate(chunks):
        if len(chunk) >= limit:
            msg = f"Chuck {index} {chunk[:20]} isn't under the chunk limit"
            raise Exception(msg)

    return chunks
