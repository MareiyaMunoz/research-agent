from urllib.parse import urlparse


class SourceLedger:
    """Records every page the agent actually read, in reading order.

    The ledger is the source of truth for citations: the report's reference
    list is generated from here, never from what the model claims it read.
    """

    def __init__(self):
        self._sources = []   # list of dicts: {"n": 1, "url": ..., "title": ..., "step": ...}
        self._by_url = {}    # url -> index in _sources, for duplicate detection

    def record(self, url: str, step: int, title: str | None = None) -> int:
        """Register a successfully read page. Returns its citation number.

        Reading the same URL twice returns the original number instead of
        creating a duplicate entry.
        """
        url = url.strip()
        if url in self._by_url:
            return self._sources[self._by_url[url]]["n"]

        source = {
            "n": len(self._sources) + 1,   # citations are 1-based: [1], [2], ...
            "url": url,
            "title": title or _fallback_title(url),
            "step": step,
        }
        self._by_url[url] = len(self._sources)
        self._sources.append(source)
        return source["n"]

    def get(self, n: int) -> dict | None:
        """Look up a source by its citation number."""
        if 1 <= n <= len(self._sources):
            return self._sources[n - 1]   # [1] lives at index 0
        return None

    def urls(self) -> list[str]:
        return [s["url"] for s in self._sources]

    def as_list(self) -> list[dict]:
        """A copy of the sources, safe to hand to callers."""
        return [dict(s) for s in self._sources]

    def __len__(self) -> int:
        return len(self._sources)


def _fallback_title(url: str) -> str:
    """When we have no title from search results, use the domain."""
    host = urlparse(url).netloc
    return host.removeprefix("www.") or url
