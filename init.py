import json
import os
from dotenv import load_dotenv
from graphAPI.graphs import getNotebookList, getSectionList, getPagesList


with open('cache.json') as f:
    CACHE = json.load(f)


def updateCache():
    with open('cache.json', 'w') as f:
        json.dump(CACHE, f, indent=4)


def main():

    update_cache = False
    if "Notebooks" in CACHE and CACHE["Notebooks"]:
        notebook_list = CACHE["Notebooks"]
    else:
        notebook_list = getNotebookList()
        CACHE["Notebooks"] = notebook_list
        update_cache = True

    notebook_name = os.getenv("NOTEBOOK_NAME")

    if notebook_name not in notebook_list:
        raise Exception("Notebook not found in the OneNote directory !")

    if "Sections" in CACHE and CACHE["Sections"]:
        sections = CACHE["Sections"]
    else:
        sections = getSectionList(notebook_list[notebook_name]["id"])
        CACHE["Sections"] = sections
        update_cache = True

    section_name = os.getenv("SECTION_NAME")
    if section_name not in sections:
        raise Exception(f"Section not found in the OneNote {notebook_name} directory !")

    if "Pages" in CACHE and CACHE["Pages"] and section_name in CACHE["Pages"]:
        pages = CACHE["Pages"][section_name]
    else:
        pages = getPagesList(sections[section_name]["id"])
        CACHE["Pages"] = {section_name: pages}
        update_cache = True

    for p in pages:
        print(p['title'], p['contentUrl'])

    if update_cache:
        updateCache()


if __name__ == "__main__":
    main()



