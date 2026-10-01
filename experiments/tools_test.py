from app.agent.tools import read_page, search

if __name__ == "__main__":
    results = search("PostgreSQL vs MongoDB for small applications")
    for r in results["results"]:
        print(r["title"], "->", r["url"])

    page = read_page(results["results"][0]["url"])
    if "error" in page:
        print("ERROR:", page["error"])
    else:
        print(page["text"][:500])
        print("truncated:", page["truncated"], "| total chars:", page["total_chars"])