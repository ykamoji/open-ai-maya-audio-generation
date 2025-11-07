import os
import json
import time
from graphAPI.graphs import getContent
from OIGenerator.utils import content_stats
from dotenv import load_dotenv

load_dotenv(override=True)

if "GRAPH_ACCESS_TOKEN" not in os.environ:
    raise Exception("Load Graph Access token first !")

access_token = os.getenv("GRAPH_ACCESS_TOKEN")

with open('cache.json') as f:
    CACHE = json.load(f)

with open('contentCache.json') as f:
    CONTENT_CACHE = json.load(f)


def updateCache():
    with open('contentCache.json', 'w') as f:
        json.dump(CONTENT_CACHE, f)

def main():
    update_cache = False
    notebook_name = os.getenv("NOTEBOOK_NAME")
    section_name = os.getenv("SECTION_NAME")
    print(f"Running voice generation for {notebook_name} {section_name}")

    pages = CACHE["Pages"][section_name][:1]

    for page in pages:
        if CONTENT_CACHE and page["title"] in CONTENT_CACHE:
            content = CONTENT_CACHE[page["title"]]
        else:
            content = getContent(page["id"])
            CONTENT_CACHE[page["title"]] = content
            update_cache = True
        print(f"Downloaded the content {page['title']} : {content_stats(content)}")

    if update_cache: updateCache()



if __name__ == '__main__':
    main()