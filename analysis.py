import json
import re
from collections import defaultdict

def analysis():

    metrics = ["count", "avg", "max", "min"]
    params = ["characters", "words", "lines", "paragraphs"]

    stats = {}
    for param in params:
        stats[param] = {}
        for metric in metrics:
            stats[param][metric] = 0

    with open('contentCache.json') as f:
        pages = json.load(f)

    mx, mn = {}, {}
    for param in params:
        mx[param] = 0
        mn[param] = 1e10

    for page, data in pages.items():
        for param in params:
            stats[param]["count"] += data["stats"][param]

            if len(pages) > 1:
                if mx[param] < data["stats"][param]:
                    mx[param] = data["stats"][param]
                    mx["title"] = page

                if mn[param] > data["stats"][param]:
                    mn[param] = data["stats"][param]
                    mn["title"] = page

    for param in params:
        stats[param]["avg"] = stats[param]["count"] / len(pages)

        if len(pages) > 1:
            stats[param]["max"] = mx[param]
            stats[param]["min"] = mn[param]
            stats[param]["min_title"] = mn["title"]
            stats[param]["max_title"] = mx["title"]

    print("Statistics:")
    for param in params:
        print(f"\t{param}:")
        for metric in metrics:
            extra = ""
            if len(pages) > 1:
                if metric == "max":
                    extra = f"""({stats[param]["max_title"]})"""
                if metric == "min":
                    extra = f"""({stats[param]["min_title"]})"""
            elif metric == "max" or metric == "min":
                continue
            print(f"\t\t{metric}: {stats[param][metric]:.2f} {extra}")
    print("\n\n")


def post_processing_helper():
    with open('voiceCache.json') as f:
        content = json.load(f)

    count = 0
    total = 0
    to_remove = 0
    for key in content:
        total += len(content[key])
        for paras in content[key]:
            if "\n\n" in paras:
                count += 1
                for para in paras.split("\n\n"):
                    if (para.startswith("I made") and ("adjustments" in para or "changes" in para)) or "paragraph" in para.lower():
                        to_remove += 1
                        print(para)

    print(f"\n\n{(count / total)*100:.2f} % of original paragraphs were split further into smaller paragraphs")
    print(f"\n\n{to_remove} paragraphs need to be removed from original paragraphs")




if __name__ == "__main__":
    # analysis()
    post_processing_helper()
