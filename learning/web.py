"""
web.py — PY-V (learning/)
V's window to the internet (Phase 13): search, then read a page as plain text.
Only search words and page addresses leave the laptop — never chat history,
code or memory (owner's rule; the callers build the words, see build_query).

  search(query)      DuckDuckGo's HTML page (no key, no account); Wikipedia's
                     search as a fallback when DuckDuckGo gives nothing
  read_page(url)     the page's readable text: headings, paragraphs, list items
                     and code, without menus, scripts and footers

Polite: one request at a time, a pause between them, a clear User-Agent, short
timeouts. No internet (or Kaggle with internet off) → empty results, never a crash.
Standard library HTML parsing only (html.parser) — nothing to install.
"""

import html
import logging
import re
import threading
import time
from html.parser import HTMLParser
from typing import List, Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) V-local-assistant/1.0 (personal study; one request at a time)"
TIMEOUT    = 15
PAUSE      = 1.5          # seconds between requests
MAX_BYTES  = 2_000_000    # a page bigger than this is cut
SKIP_HOSTS = ("youtube.com", "youtu.be", "facebook.com", "instagram.com", "tiktok.com", "twitter.com", "x.com",
              "linkedin.com", "pinterest.", "reddit.com/login")

_lock = threading.Lock()
_last = [0.0]


def _get(url: str, **kwargs) -> Optional[requests.Response]:
    with _lock:
        wait = PAUSE - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            res = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"}, timeout=TIMEOUT,
                               stream=True, **kwargs)
            res._content = res.raw.read(MAX_BYTES, decode_content=True)   # never more than MAX_BYTES
            res._content_consumed = True
            return res
        except Exception as e:
            logger.info(f"web: {url[:80]} failed ({e})")
            return None
        finally:
            _last[0] = time.time()


def online() -> bool:
    """Can V reach the internet at all? (one quick request)"""
    res = _get("https://duckduckgo.com/")
    return res is not None and res.status_code < 500


# ─── search ──────────────────────────────────────────────────────────────────

_RESULT  = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SNIPPET = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAGS    = re.compile(r"<[^>]+>")


def search(query: str, max_results: int = 6) -> List[dict]:
    """[{"title", "url", "snippet"}] — best first; [] when offline or nothing found."""
    query = query.strip()
    if not query:
        return []
    found = _duckduckgo(query, max_results)
    if not found:
        found = _wikipedia(query, max_results)
    return found


def _duckduckgo(query: str, max_results: int) -> List[dict]:
    res = _get(f"https://html.duckduckgo.com/html/?q={quote(query)}")
    if res is None or res.status_code != 200:
        return []
    page     = res.content.decode("utf-8", errors="replace")
    snippets = [_text(s) for s in _SNIPPET.findall(page)]
    results  = []
    for i, (href, title) in enumerate(_RESULT.findall(page)):
        url = _real_url(href)
        if not url or any(h in url for h in SKIP_HOSTS) or "duckduckgo.com/y.js" in url:
            continue
        results.append({"title": _text(title), "url": url, "snippet": snippets[i] if i < len(snippets) else ""})
        if len(results) >= max_results:
            break
    return results


def _wikipedia(query: str, max_results: int) -> List[dict]:
    res = _get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": max_results})
    if res is None or res.status_code != 200:
        return []
    try:
        hits = res.json()["query"]["search"]
    except Exception:
        return []
    return [{"title": h["title"], "url": "https://en.wikipedia.org/wiki/" + quote(h["title"].replace(" ", "_")),
             "snippet": _text(h.get("snippet", ""))} for h in hits]


def _real_url(href: str) -> Optional[str]:
    """DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<the real address>."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(target) if target else None
    return href if parsed.scheme in ("http", "https") else None


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub("", fragment))).strip()


# ─── reading a page ──────────────────────────────────────────────────────────

class _Reader(HTMLParser):
    """Collects readable text; skips scripts, styles, menus, headers, footers, forms."""
    SKIP  = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button", "select",
             "iframe", "template"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "pre", "tr", "dt", "dd", "blockquote", "br", "div", "section",
             "article", "table", "td", "th"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip, self.pre, self.title, self._in_title = [], 0, 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "pre":
            self.pre += 1
            self.parts.append("\n```\n")
        elif tag in ("h1", "h2", "h3", "h4"):
            self.parts.append("\n\n## ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.parts.append("\n```\n")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data if self.pre else re.sub(r"\s+", " ", data))


def read_page(url: str, max_chars: int = 20000) -> Optional[dict]:
    """{"url", "title", "text"} — the page's readable text, or None (offline, not HTML, too little text)."""
    res = _get(url)
    if res is None or res.status_code != 200:
        return None
    kind = res.headers.get("Content-Type", "")
    if "html" not in kind and "text/plain" not in kind:
        return None
    raw = res.content.decode(res.encoding or "utf-8", errors="replace")
    if "text/plain" in kind:
        title, text = url.rsplit("/", 1)[-1], raw
    else:
        reader = _Reader()
        try:
            reader.feed(raw)
        except Exception:
            return None
        title, text = reader.title.strip(), "".join(reader.parts)
    lines = [l.rstrip() for l in text.split("\n")]
    text  = re.sub(r"\n{3,}", "\n\n", "\n".join(l for l in lines if l.strip() or l == "")).strip()
    if len(text) < 300:
        return None
    return {"url": url, "title": re.sub(r"\s+", " ", title)[:150] or url, "text": text[:max_chars]}


def best_part(text: str, words: List[str], max_chars: int = 3000) -> str:
    """The paragraphs of a page that share the most words with the question / topic, in page order, within max_chars."""
    paras  = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
    if not paras:
        return text[:max_chars]
    wanted = {w.lower() for w in words if len(w) > 2}
    scored = []
    for i, p in enumerate(paras):
        low   = p.lower()
        score = sum(low.count(w) for w in wanted) + (0.5 if i < 3 else 0)
        scored.append((score, i))
    chosen, used = [], 0
    for score, i in sorted(scored, key=lambda s: (-s[0], s[1])):
        if score <= 0 and chosen:
            break
        if used + len(paras[i]) > max_chars:
            if not chosen:
                chosen.append(i)
            continue
        chosen.append(i)
        used += len(paras[i]) + 2
    return "\n\n".join(paras[i] for i in sorted(chosen))[:max_chars]
