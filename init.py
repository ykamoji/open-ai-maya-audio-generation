import json
import os
from dotenv import load_dotenv
from graphAPI.graphs import getNotebookList


with open('cache.json') as f:
    CACHE = json.load(f)


def updateCache():
    with open('cache.json', 'w') as f:
        json.dump(CACHE, f, indent=4)


def main():

    update_cache = False
    if "Notebook" in CACHE and CACHE["Notebook"]:
        notebook_list = CACHE["Notebook"]
    else:
        notebook_list = getNotebookList()
        CACHE["Notebook"] = notebook_list
        update_cache = True

    notebook_name = os.getenv("NOTEBOOK_NAME")

    if notebook_name not in notebook_list:
        raise Exception("Notebook not found in the OneNote directory !")



    if update_cache:
        updateCache()


if __name__ == "__main__":
    main()



