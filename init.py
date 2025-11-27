import json
import argparse
import yaml
import time
from analysis import analysis
from utils import CustomObject, get_yaml_loader, updateCache, create_or_load_Cache, content_stats, getChapterNo
from GraphAPI.graphs import GraphAPI


class Initialization:

    def __init__(self):

        self.CACHE = create_or_load_Cache('cache/cache.json')
        self.CONTENT_CACHE = create_or_load_Cache('cache/contentCache.json')

        with open('default.yaml', 'r') as file:
            config = yaml.load(file, get_yaml_loader())

        x = json.dumps(config)
        self.Args = json.loads(x, object_hook=lambda d: CustomObject(**d))

        self.graph = GraphAPI(self.Args.Graph)
        self.notebook_name = self.Args.Graph.NotebookName
        self.section_name = self.Args.Graph.SectionName

        parser = argparse.ArgumentParser(description="Load Data")
        parser.add_argument("--refreshPages", type=bool, default=False, help="ContentReload")
        parser.add_argument("--pageLimit", type=int, default=None, help="ContentReload")
        args = parser.parse_args()
        self.Args.Graph.RefreshPages = args.refreshPages
        self.Args.Graph.PageLimit = args.pageLimit

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

        if update_cache: updateCache('cache/cache.json', self.CACHE)

        print(f"Found {len(pages)} pages for section {self.section_name} in notebook {self.notebook_name}")

        limit = len(pages)
        if self.Args.Graph.PageLimit:
            limit = self.Args.Graph.PageLimit

        print("Running Page Content API")
        nb = self.CONTENT_CACHE.setdefault(self.notebook_name, {})
        section = nb.setdefault(self.section_name, {})
        progress = 0
        for page in pages[:limit]:
            update_cache = False
            if self.Args.Graph.RefreshPages or not page["title"] in section:
                page_content = self.graph.getContent(page["id"])
                section[page["title"]] = {
                    "content": page_content,
                    "stats": content_stats(page_content),
                }
                update_cache = True

            if update_cache:
                section = dict(sorted(section.items(), key=lambda x: getChapterNo(x[0])))
                self.CONTENT_CACHE[self.notebook_name][self.section_name] = section
                updateCache('cache/contentCache.json', self.CONTENT_CACHE)
                print(f"Downloaded the page {page['title']}")
                time.sleep(1)

            progress += 1
            if progress >= limit:
                break


        print(f"Downloaded the {progress} page(s)")

        print("Completed Initialization")
        analysis(self.notebook_name, self.section_name)


if __name__ == "__main__":
    Initialization().run()
