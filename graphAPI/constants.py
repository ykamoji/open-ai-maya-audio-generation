GRAPH_API = "https://graph.microsoft.com/v1.0/me/onenote"

SELECTORS = "?$select=id,createdDateTime,displayName,lastModifiedDateTime"

NOTEBOOK_LIST_URL = GRAPH_API + "/notebooks" + SELECTORS

SECTION_LIST_URL = GRAPH_API + "/notebooks/{}/sections" + SELECTORS

PAGE_LIST_SIZE = 10

PAGES_LIST_SELECTORS = f"?$select=id,title,contentUrl,createdDateTime,lastModifiedDateTime&$count=true&$top={PAGE_LIST_SIZE}"

PAGES_LIST_URL = GRAPH_API + "/sections/{}/pages" + PAGES_LIST_SELECTORS
