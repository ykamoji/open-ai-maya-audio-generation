import re
from tqdm import tqdm


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
                    if p.strip().startswith("Note:") or p.strip().startswith("I made") or p.strip().startswith("Changes made:"):
                        parts = parts[:i]
                        clean_paragraphs_count += 1
                        break
                paragraph = "\n\n".join([part.strip() for part in parts])

            if "\n\n" in paragraph:
                split_paragraph = True
                split_paragraphs_count += 1

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
