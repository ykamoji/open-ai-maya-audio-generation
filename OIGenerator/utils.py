
def content_stats(content):
    count = len(content.replace(" ", "").replace("\n", ""))
    word_count = len(content.split())
    lines = len(content.splitlines())
    return f"{count} characters, {word_count} words, {lines} lines"
