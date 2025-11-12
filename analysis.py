import json


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

            print(f"\t\t{metric}: {stats[param][metric]:.2f} {extra}")
    print("\n\n")


