from pydub import AudioSegment


def content_stats(content):
    count = len(content.replace(" ", "").replace("\n", ""))
    word_count = len(content.split())
    lines = len(content.split('.'))
    paras = len(content.split("\n\n"))
    return f"{count} characters, {word_count} words, {lines} lines, {paras} paragraphs"


def createChunks(content, limit):

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        # If single paragraph longer than max_chars, break it manually
        if len(para) > limit:
            # Split that paragraph into smaller sub-chunks
            for i in range(0, len(para), limit):
                sub_chunk = para[i:i + limit]
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sub_chunk.strip())
            continue

        # If adding this paragraph exceeds limit, start new chunk
        if len(current_chunk) + len(para) + 2 > limit:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

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