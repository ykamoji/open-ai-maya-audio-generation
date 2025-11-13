import json
import os
import yaml

from analysis import analysis
from utils import CustomObject, get_yaml_loader, updateCache, createCache, content_stats
from GraphAPI.graphs import GraphAPI


class Initialization:

    def __init__(self):

        self.CACHE = createCache('cache.json')
        self.CONTENT_CACHE = createCache('contentCache.json')

        with open('default.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        self.graph = GraphAPI(self.Args.Graph)
        self.notebook_name = self.Args.Graph.NotebookName
        self.section_name = self.Args.Graph.SectionName

    def run(self):

        update_cache = False
        if "Notebooks" in self.CACHE and self.CACHE["Notebooks"]:
            notebook_list = self.CACHE["Notebooks"]
        else:
            notebook_list = self.graph.getNotebookList()
            self.CACHE["Notebooks"] = notebook_list
            update_cache = True

        if self.notebook_name not in notebook_list:
            raise Exception("Notebook not found in the OneNote directory !")

        if "Sections" in self.CACHE and self.CACHE["Sections"]:
            sections = self.CACHE["Sections"]
        else:
            sections = self.graph.getSectionList(notebook_list[self.notebook_name]["id"])
            self.CACHE["Sections"] = sections
            update_cache = True

        if self.section_name not in sections:
            raise Exception(f"Section not found in the OneNote {self.notebook_name} directory !")

        if not self.Args.Graph.RefreshPages and "Pages" in self.CACHE \
                and self.CACHE["Pages"] and self.section_name in self.CACHE["Pages"]:
            pages = self.CACHE["Pages"][self.section_name]
        else:
            pages = self.graph.getPagesList(sections[self.section_name]["id"])
            self.CACHE["Pages"] = {self.section_name: pages}
            update_cache = True

        if update_cache: updateCache('cache.json', self.CACHE)

        print(f"Found {len(pages)} pages for section {self.section_name} in notebook {self.notebook_name}")

        limit = len(pages)
        if self.Args.Graph.PageLimit:
            limit = self.Args.Graph.PageLimit

        for page in pages[:limit]:
            update_cache = False
            if not self.Args.Graph.RefreshPages and page["title"] in self.CONTENT_CACHE:
                page_content = self.CONTENT_CACHE[page["title"]]
            else:
                page_content = self.graph.getContent(page["id"])
                self.CONTENT_CACHE[page["title"]] = {
                    "content": page_content,
                    "stats": content_stats(page_content),
                }
                update_cache = True
                print(f"Downloaded the page {page['title']}")

            if update_cache: updateCache('contentCache.json', self.CONTENT_CACHE)

        print("Completed Initialization")
        analysis()


if __name__ == "__main__":
    Initialization().run()
