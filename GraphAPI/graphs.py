import os
from collections import defaultdict
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import json
import GraphAPI.constants as C


class GraphAPI:

    def __init__(self, Args):
        self.Args = Args

        load_dotenv(Args.EnvPath, override=True)

        if "GRAPH_ACCESS_TOKEN" not in os.environ:
            raise Exception("Load Graph Access token first !")

        access_token = os.getenv("GRAPH_ACCESS_TOKEN")

        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

    def commonGetAPI(self, api_name, url):

        print(f"Running {api_name}")
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"{api_name} Failed. {response.status_code} {response.text}")

        json_response = json.loads(response.text)

        value_list = defaultdict(dict)
        if "value" in json_response:
            for value in json_response["value"]:
                value_list[value["displayName"]] = value

        return value_list

    def getNotebookList(self):
        return self.commonGetAPI("Notebook List API", C.NOTEBOOK_LIST_URL)

    def getSectionList(self, notebook_id):
        return self.commonGetAPI("Section List API", C.SECTION_LIST_URL.replace("{}", notebook_id))

    def threadPages(self, section_id, skip):
        response = requests.get(C.PAGES_LIST_URL.replace("{}", section_id) + f"&skip={skip}", headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Unable to get pages list {response.status_code} {response.text}")

        response = json.loads(response.text)
        return response["value"]

    def getPagesList(self, section_id):
        print("Running Pages List API")

        response = requests.get(C.PAGES_LIST_URL.replace("{}", section_id), headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Unable to get pages list {response.status_code} {response.text}")

        response = json.loads(response.text)

        pages_list = []
        if "value" in response:
            pages_list = response["value"]

        total_pages = response["@odata.count"]

        pagination = range(C.PAGE_LIST_SIZE, total_pages, C.PAGE_LIST_SIZE)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.threadPages, section_id, skip) for skip in pagination]

            for f in as_completed(futures):
                pages_list.extend(f.result())

        pages_list = sorted(pages_list, key=lambda x: datetime.fromisoformat(x["createdDateTime"].replace("Z", "+00:00")))

        return pages_list

    def getContent(self, page_id):
        print("Running Page Content API")

        self.headers["Content-Type"] = "text/html"
        response = requests.get(C.CONTENT_URL.replace("{}", page_id), headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Unable to get page content {response.status_code} {response.text}")

        html_response = response.text
        soup = BeautifulSoup(html_response, "html.parser")
        body = soup.find("body")
        text = body.get_text()

        return text

