import json
import os
import yaml
from utils import CustomObject, get_yaml_loader, updateCache
from GraphAPI.graphs import GraphAPI


def main():

    if os.path.isfile('cache.json'):
        with open('cache.json') as f:
            CACHE = json.load(f)
    else:
        CACHE = {}
        with open('cache.json', 'w') as f:
            json.dump(CACHE, f)

    with open('config.yaml', 'r') as file:
        config = yaml.load(file, get_yaml_loader())

    x = json.dumps(config)
    Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

    graph = GraphAPI(Args.Graph)

    update_cache = False
    if "Notebooks" in CACHE and CACHE["Notebooks"]:
        notebook_list = CACHE["Notebooks"]
    else:
        notebook_list = graph.getNotebookList()
        CACHE["Notebooks"] = notebook_list
        update_cache = True

    notebook_name = Args.Graph.NotebookName

    if notebook_name not in notebook_list:
        raise Exception("Notebook not found in the OneNote directory !")

    if "Sections" in CACHE and CACHE["Sections"]:
        sections = CACHE["Sections"]
    else:
        sections = graph.getSectionList(notebook_list[notebook_name]["id"])
        CACHE["Sections"] = sections
        update_cache = True

    section_name = Args.Graph.SectionName

    if section_name not in sections:
        raise Exception(f"Section not found in the OneNote {notebook_name} directory !")

    if not Args.Graph.RefreshPages and "Pages" in CACHE and CACHE["Pages"] and section_name in CACHE["Pages"]:
        pages = CACHE["Pages"][section_name]
    else:
        pages = graph.getPagesList(sections[section_name]["id"])
        CACHE["Pages"] = {section_name: pages}
        update_cache = True

    if update_cache:
        updateCache('cache.json', CACHE)

    print(f"Found {len(pages)} pages")

    print("Completed Initialization")


if __name__ == "__main__":
    main()



