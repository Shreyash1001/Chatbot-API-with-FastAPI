from serpapi import GoogleSearch
import os

SERPAPI_KEY = os.getenv("SERPAPI_KEY")  

def web_search(query):
    if not SERPAPI_KEY:
        return "No API Key Found."
    
    search = GoogleSearch({
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 3
    })
    
    results = search.get_dict().get("organic_results", [])
    return " ".join([res["snippet"] for res in results])

query = "Latest AI trends"
print(web_search(query))
