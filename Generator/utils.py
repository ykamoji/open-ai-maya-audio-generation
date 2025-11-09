from pydub import AudioSegment


def content_stats(content):
    count = len(content.replace(" ", "").replace("\n", ""))
    word_count = len(content.split())
    lines = len(content.split('.'))
    paras = len(content.split("\n\n"))
    return f"{count} characters, {word_count} words, {lines} lines, {paras} paragraphs"


def createChunks(content, limit=None):

    chunks = []
    paragraphs = [p.strip("\n").strip() for p in content.split("\n\n") if p.strip()]
    if limit is not None:
        for para in paragraphs:
            if len(para) >= limit:
                lines = [line for line in para.split(".") if line.strip()]
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

    else:
        for para in paragraphs:
            lines = [line.strip().replace('.','') + '.' for line in para.split(". ") if line.strip()]
            chunks.extend(lines)
            chunks.append('')

    return chunks


def merge_audio(files, output_file, format="mp3"):
    combined = AudioSegment.empty()
    for file in files:
        print(f"Merging {file}...")
        segment = None
        if format == "mp3":
            segment = AudioSegment.from_mp3(file)
        elif format == "wav":
            segment = AudioSegment.from_wav(file)

        combined += segment

    combined.export(output_file, format=format)

    print(f"Merged audio saved as {output_file}")
