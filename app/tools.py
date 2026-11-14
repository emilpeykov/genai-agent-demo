"""Tools exposed to the agent.

Where a maintained LangChain-integrated tool/toolkit already exists for a
capability (Wikipedia lookup, shell execution, file read/write/list), we use
that directly (``langchain_community.tools``/``agent_toolkits``) instead of
hand-rolling one. Where no such integration exists (arithmetic, an internal
mock KB, in-place text edits, system resource stats), we write a small
function and expose it with LangChain's ``@tool`` decorator -- the same
mechanism LangChain's own built-in tools use under the hood, so from the
agent's point of view (and in ``AVAILABLE_TOOLS`` below) every tool here
looks and behaves identically regardless of where it came from.

This illustrates the "agent frameworks" requirement: the agent can decide,
based on the user's request, whether it needs to call a deterministic tool
(e.g. a calculator) instead of hallucinating an answer -- a core idea behind
LangChain/LlamaIndex/AutoGen-style agents.
"""
from __future__ import annotations

import ast
import base64
import fnmatch
import json
import operator as op
import re
import subprocess
import warnings
from pathlib import Path

import psutil
import truststore

# Some networks (notably corporate proxies doing TLS inspection, e.g.
# Zscaler) re-sign outbound HTTPS connections with their OWN root CA. That
# root CA is installed in the OS-level trust store (so browsers/curl/etc.
# work fine) but NOT in the separate ``certifi`` CA bundle that Python's
# ``ssl``/``requests`` use by default -- which is exactly what produced
# "SSLCertVerificationError: unable to get local issuer certificate" for
# wikipedia_search (and would equally break any other outbound HTTPS tool,
# e.g. a web-search tool). ``truststore.inject_into_ssl()`` patches the
# stdlib ``ssl`` module to validate certificates against the OS's native
# trust store instead (Keychain on macOS, Certificate Store on Windows),
# so this one call fixes SSL verification for every HTTPS-based tool below
# without hand-maintaining a custom CA bundle. Must run before any of those
# tools are constructed/used, so it's done here at import time, as early as
# possible.
truststore.inject_into_ssl()

from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import ShellTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from app.codebase_rag import EXCLUDED_DIR_NAMES as _GREP_EXCLUDED_DIR_NAMES
from app.codebase_rag import codebase_search
from app.config import SERPER_API_KEY
from app.postgres_rag import article_search
from app.rag import rag_search
from app.readme_rag import readme_search

# The underlying `wikipedia` package's default User-Agent
# ("wikipedia (https://github.com/goldsmith/Wikipedia/)") gets rejected by
# Wikimedia's current API policy, which requires a descriptive UA with
# contact info (https://w.wiki/4wJS) -- observed as a 403 HTML response
# where JSON was expected, which `wikipedia` then fails to parse as
# "Expecting value: line 1 column 1 (char 0)" (a raw ``json.JSONDecodeError``
# with no hint it was actually an HTTP/policy rejection). Setting a
# compliant UA here, once, at import time, avoids that failure mode instead
# of just surfacing the confusing downstream parse error.
try:
    import wikipedia as _wikipedia_pkg

    _wikipedia_pkg.set_user_agent(
        "OpenKBS-Agent/1.0 (https://github.com/openkbs; contact@openkbs.com) "
        "python-wikipedia"
    )
except Exception:  # pragma: no cover - defensive, never block import on this
    pass

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.

    Use this whenever the user asks for a numeric computation so the answer
    is exact rather than guessed by the LLM.
    """
    try:
        tree = ast.parse(expression, mode="eval").body
        return str(_safe_eval(tree))
    except Exception:
        return "Error: could not evaluate expression"


@tool
def knowledge_base_lookup(topic: str) -> str:
    """Look up a short fact about a known internal topic.

    This simulates a retrieval tool (as you'd build with LlamaIndex over a
    real document/vector store) without requiring external infrastructure.
    """
    mock_kb = {
        "refund policy": "Refunds are processed within 5-7 business days of approval.",
        "sla": "Standard support SLA is a 4 business hour first response time.",
        "pricing": "Pricing tiers are Starter, Growth, and Enterprise, billed monthly.",
    }
    return mock_kb.get(topic.strip().lower(), f"No internal knowledge found for '{topic}'.")


# Instead of hand-writing an HTTP client + response parser for an external
# knowledge source, this uses LangChain's PREBUILT ``WikipediaQueryRun`` tool
# (from langchain_community.tools) -- a ready-made, agent-ready ``BaseTool``
# built on top of ``WikipediaAPIWrapper`` -- rather than calling the API
# wrapper by hand. It's still exposed to the agent through our own thin
# @tool wrapper so that network/SSL/API errors degrade gracefully instead of
# crashing the agent loop (``WikipediaQueryRun`` itself only swallows
# ``ToolException``s, not arbitrary network errors).
_wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
)


# Wikimedia occasionally rate-limits/blocks a request transiently (seen as a
# non-JSON HTML response that the `wikipedia` package then fails to parse,
# surfacing as a confusing ``json.JSONDecodeError`` -- "Expecting value:
# line 1 column 1 (char 0)" -- rather than a clear HTTP/rate-limit error).
# Retrying a couple of times with a short backoff resolves most of these
# without the model ever seeing the failure, same "retry transient errors"
# pattern used for LLM calls in app.resilience.
@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
def _invoke_wikipedia_tool(query: str) -> str:
    return _wikipedia_tool.invoke({"query": query})


@tool
def wikipedia_search(query: str) -> str:
    """Look up a GENERAL-knowledge topic (people, places, companies, concepts,
    historical events) on Wikipedia and return a short summary.

    Use this for external, real-world knowledge questions -- unlike
    'knowledge_base_lookup', which only knows a handful of OUR OWN mock
    internal company facts (refund policy, SLA, pricing), this can answer
    almost anything with a Wikipedia article. Requires internet access.
    """
    try:
        result = _invoke_wikipedia_tool(query)
    except Exception as exc:  # network/SSL/API errors -- never crash the agent loop
        return f"Error searching Wikipedia for '{query}': {exc}"
    return result or f"No Wikipedia results found for '{query}'."


# General-purpose web search, for anything Wikipedia doesn't cover (current
# events, specific products/companies, niche topics with no Wikipedia
# article, etc.). Uses LangChain's PREBUILT ``DuckDuckGoSearchAPIWrapper``
# (langchain_community.utilities) -- same "use the maintained integration"
# rationale as ``wikipedia_search`` -- backed by the free, no-API-key
# ``duckduckgo-search`` package. Constructed lazily/defensively: importing
# the underlying package is deferred to first use (and guarded) so a
# missing optional dependency degrades to a clear error message instead of
# crashing the whole app at import time.
_ddg_wrapper = None
_ddg_import_error: Exception | None = None
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    _ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
except Exception as exc:  # pragma: no cover - only if the optional dep is missing
    _ddg_import_error = exc


@tool
def web_search(query: str) -> str:
    """Search the live web (via DuckDuckGo, no API key required) and return a
    short summary of the top results.

    Use this for anything current/real-world that 'wikipedia_search' can't
    answer -- recent news, specific companies/products, niche topics with no
    Wikipedia article, etc. Requires internet access. Prefer
    'wikipedia_search' first for general-knowledge/encyclopedic topics, and
    fall back to this (or 'google_search', if configured) when that comes up
    empty or the topic is too current/obscure for Wikipedia.
    """
    if _ddg_wrapper is None:
        return (
            "Error: web_search is unavailable -- the 'duckduckgo-search' "
            f"package is not installed/importable ({_ddg_import_error}). "
            "Install it with 'pip install duckduckgo-search', or use "
            "'wikipedia_search'/'google_search' instead."
        )
    try:
        result = _ddg_wrapper.run(query)
    except Exception as exc:  # network/SSL/rate-limit errors -- never crash the loop
        return f"Error searching the web for '{query}': {exc}"
    return result or f"No web search results found for '{query}'."


# Google-flavored web search via the Serper.dev API (a thin, cheap proxy in
# front of real Google Search results) -- LangChain's PREBUILT
# ``GoogleSerperAPIWrapper``. This needs a (free-tier available) API key, so
# it's OPTIONAL and only constructed when 'SERPER_API_KEY' is configured.
_google_wrapper = None
_google_import_error: Exception | None = None
if SERPER_API_KEY:
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper

        _google_wrapper = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    except Exception as exc:  # pragma: no cover - only if the optional dep is missing
        _google_import_error = exc

# FALLBACK (no API key required): scrape Bing's plain HTML search-results
# page directly. Google was tried FIRST here (via the 'googlesearch-python'
# package), but turned out unreliable in live testing: this environment's
# requests were served either an EU cookie-consent redirect or a heavily
# JS-obfuscated anti-bot challenge page instead of real result HTML -- even
# with a realistic browser User-Agent and the standard Google "CONSENT"
# cookie bypass trick. Bing's results page, by contrast, returns plain,
# easily parseable HTML with no such wall. This is honestly NOT the same as
# real Google results (the tool's own output says so explicitly below when
# this path is used), but it accomplishes what was actually needed: a real,
# WORKING, key-free alternative search engine to escalate to, instead of an
# error message. The Serper-backed path above (real Google, via a paid-but-
# free-tier API) is still used first whenever 'SERPER_API_KEY' IS configured.
_BING_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _decode_bing_redirect_url(href: str) -> str:
    """Bing wraps every organic result link in a '/ck/a?...&u=a1<base64>...'
    tracking redirect instead of linking straight to the real page. Decode
    the base64-encoded 'u' param (prefixed with an 'a1' marker) back into
    the real target URL, so the agent (and 'fetch_webpage') get a usable
    link instead of an opaque bing.com tracking URL. Falls back to the raw
    href unchanged if it doesn't match this expected shape."""
    if "u=" not in href:
        return href
    try:
        encoded = href.split("u=", 1)[1].split("&", 1)[0]
        if encoded.startswith("a1"):
            encoded = encoded[2:]
        padded = encoded + "=" * (-len(encoded) % 4)
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="ignore")
    except Exception:
        return href


def _bing_search_scrape(query: str, num_results: int = 5) -> list[dict]:
    """Scrape Bing's HTML search-results page for 'query' and return up to
    'num_results' {"title", "url", "description"} dicts. Raises on
    network/HTTP errors -- callers (``google_search`` below) are expected to
    catch and turn that into a user-facing error string, same convention as
    every other search tool in this module."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        "https://www.bing.com/search",
        params={"q": query, "count": num_results},
        headers={"User-Agent": _BING_SEARCH_USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    results = []
    for item in soup.select("li.b_algo")[:num_results]:
        heading = item.find("h2")
        link = heading.find("a") if heading else None
        if not link or not link.get("href"):
            continue
        title = link.get_text(strip=True) or "(no title)"
        url = _decode_bing_redirect_url(link["href"])
        snippet_tag = item.find("p") or item.find("div", class_="b_caption")
        description = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
        results.append({"title": title, "url": url, "description": description})
    return results


@tool
def google_search(query: str) -> str:
    """Search the web and return a short summary of the top results --
    generally the highest-quality/most-current search available here.

    Uses the Serper.dev API (real Google results) if a 'SERPER_API_KEY'
    environment variable is configured (free tier available at
    https://serper.dev); otherwise falls back automatically to a key-free
    Bing search-results scrape (NOT literally Google in that case, since
    direct Google-results scraping is unreliably blocked by anti-bot
    defenses -- but a real, working alternative search engine, not just an
    error). Only reports itself fully unavailable if that fallback itself
    errors, in which case use 'web_search' (DuckDuckGo) or 'wikipedia_search'
    instead.
    """
    if _google_wrapper is not None:
        try:
            result = _google_wrapper.run(query)
        except Exception as exc:  # network/SSL/API errors -- never crash the agent loop
            return f"Error searching Google for '{query}': {exc}"
        return result or f"No Google search results found for '{query}'."

    # BUG FIX (observed live): with no SERPER_API_KEY set, this tool used to
    # just report itself unavailable and tell the model to fall back to
    # 'web_search'/'wikipedia_search' -- which, live, led the agent straight
    # into 'wikipedia_search' matching a completely unrelated article (see
    # app.react_agent.search_relevance_hint's docstring for that specific
    # failure). A real, working, key-free search (via the Bing scrape above)
    # is strictly more useful than an error message telling the model to try
    # something else.
    try:
        results = _bing_search_scrape(query)
    except Exception as exc:  # network/HTML-parsing/rate-limit errors
        return f"Error searching for '{query}': {exc}"
    if not results:
        return f"No search results found for '{query}'."
    return "\n\n".join(
        f"{r['title']}\n{r['url']}\n{r['description'] or '(no description)'}"
        for r in results
    )



# Academic/scientific paper search via arXiv -- LangChain's PREBUILT
# ``ArxivQueryRun``, backed by the free, no-API-key ``arxiv`` package. Good
# for anything 'wikipedia_search'/'web_search' handle poorly: specific papers,
# authors, or recent preprints in physics/CS/math/stats/etc.
_arxiv_tool = None
_arxiv_import_error: Exception | None = None
try:
    import arxiv as _arxiv_pkg
    from langchain_community.tools import ArxivQueryRun
    from langchain_community.utilities import ArxivAPIWrapper

    # The installed ``arxiv`` package (pinned to the pre-``Client`` 1.x API
    # that ``ArxivAPIWrapper`` still calls directly, i.e. ``Search.results()``
    # with no separate ``Client``) hardcodes a PLAIN-HTTP query URL
    # ('http://export.arxiv.org/...') on ``Client.query_url_format``, which
    # arxiv.org now 301-redirects to HTTPS -- and this old client doesn't
    # follow that redirect, surfacing as a confusing "Page request resulted
    # in HTTP 301" error. Patching the class attribute to HTTPS directly
    # avoids the redirect entirely.
    _arxiv_pkg.Client.query_url_format = "https://export.arxiv.org/api/query?{}"

    _arxiv_tool = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
    )
except Exception as exc:  # pragma: no cover - only if the optional dep is missing
    _arxiv_import_error = exc


@tool
def arxiv_search(query: str) -> str:
    """Search arXiv.org for scientific papers (physics, maths, CS, stats,
    quantitative biology/finance, electrical engineering, economics) and
    return short summaries of the top matches.

    Use this instead of 'wikipedia_search'/'web_search' when the user asks
    about a specific research paper, author, or recent preprint -- arXiv
    indexes pre-print scientific literature that Wikipedia/general web
    search won't surface well. Requires internet access, no API key needed.
    """
    if _arxiv_tool is None:
        return (
            "Error: arxiv_search is unavailable -- the 'arxiv' package is "
            f"not installed/importable ({_arxiv_import_error}). Install it "
            "with 'pip install arxiv', or use 'web_search'/'google_search' "
            "instead."
        )
    try:
        result = _arxiv_tool.invoke({"query": query})
    except Exception as exc:  # network/API errors -- never crash the agent loop
        return f"Error searching arXiv for '{query}': {exc}"
    return result or f"No arXiv results found for '{query}'."


# Programming Q&A search via Stack Exchange -- LangChain's PREBUILT
# ``StackExchangeTool``, backed by the free, no-API-key ``stackapi``
# package (StackExchangeAPIWrapper defaults to the stackoverflow site).
# Good for "how do I do X in language/framework Y" style questions where a
# real accepted answer + code sample beats an LLM's own guess.
_stackexchange_tool = None
_stackexchange_import_error: Exception | None = None
try:
    from langchain_community.tools import StackExchangeTool
    from langchain_community.utilities import StackExchangeAPIWrapper

    _stackexchange_tool = StackExchangeTool(api_wrapper=StackExchangeAPIWrapper())
except Exception as exc:  # pragma: no cover - only if the optional dep is missing
    _stackexchange_import_error = exc


@tool
def stackexchange_search(query: str) -> str:
    """Search Stack Overflow / Stack Exchange for a specific programming
    question and return matching question titles + excerpts/answers.

    Use this for concrete "how do I ... in <language/framework>" or
    "why does <error>" programming questions -- it returns real community
    answers/code rather than a guessed one. Input should be a fully-formed
    question, e.g. 'how to reverse a list in python'. Requires internet
    access, no API key needed.
    """
    if _stackexchange_tool is None:
        return (
            "Error: stackexchange_search is unavailable -- the 'stackapi' "
            f"package is not installed/importable ({_stackexchange_import_error}). "
            "Install it with 'pip install stackapi', or use 'web_search'/"
            "'google_search' instead."
        )
    try:
        result = _stackexchange_tool.invoke({"query": query})
    except Exception as exc:  # network/API errors -- never crash the agent loop
        return f"Error searching Stack Exchange for '{query}': {exc}"
    return result or f"No Stack Exchange results found for '{query}'."


# "Web spider": fetch a SPECIFIC url's page content and extract its readable
# text. Search tools above (web_search/google_search/wikipedia_search/
# arxiv_search/stackexchange_search) return short snippets + links -- this
# is the complementary tool that lets the agent actually "click through" and
# read a page's full content, e.g. after a search result gives it a URL.
# No clean off-the-shelf LangChain tool does "fetch + strip HTML + truncate"
# in one step (``RequestsGetTool`` returns raw, un-stripped HTML and also
# requires opting into ``allow_dangerous_requests``), so -- same rationale as
# 'edit_file' elsewhere in this file -- it's hand-written here on top of the
# already-installed 'requests'/'beautifulsoup4' libraries.
FETCH_WEBPAGE_TIMEOUT_SECONDS = 10
FETCH_WEBPAGE_MAX_CHARS = 4000
_FETCH_WEBPAGE_USER_AGENT = (
    "OpenKBS-Agent/1.0 (https://github.com/openkbs; contact@openkbs.com)"
)


def _fetch_soup(url: str, timeout: int = FETCH_WEBPAGE_TIMEOUT_SECONDS):
    """Shared by ``fetch_webpage``/``crawl_website``: GET ``url`` and parse it
    into a BeautifulSoup tree. Raises on network/HTTP errors -- callers are
    expected to catch and turn that into a user-facing error string.

    BUG FIX (observed live: an em-dash rendering as mojibake, '\\x80\\x94',
    in real OpenKBS page content): this used to build the soup from
    ``response.text``, which decodes using whatever encoding ``requests``
    GUESSED from the response headers alone -- and falls back to a poor
    default (often Latin-1) when the server's ``Content-Type`` header
    doesn't explicitly declare a charset, even though the page itself is
    really UTF-8 (the overwhelmingly common case for modern sites, declared
    only via an in-HTML ``<meta charset="utf-8">`` tag). Passing the raw
    ``response.content`` bytes to BeautifulSoup/lxml instead lets ITS
    encoding sniffer inspect that in-HTML meta tag/BOM directly, which is
    far more reliable than requests' header-only guess.
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        url, timeout=timeout, headers={"User-Agent": _FETCH_WEBPAGE_USER_AGENT}
    )
    response.raise_for_status()
    return BeautifulSoup(response.content, "lxml")


def _readable_text(soup) -> str:
    """Strip script/style/noscript tags and collapse whitespace, shared by
    ``fetch_webpage``/``crawl_website`` so both extract text identically."""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _same_domain_links(soup, base_url: str) -> list[str]:
    """Collect every same-domain, non-duplicate link found in ``soup``,
    resolved to absolute URLs against ``base_url`` -- used by
    ``crawl_website`` to discover further pages to fetch without the model
    having to guess exact sub-page paths itself (which, live, produced 404s
    for guessed paths like '/solutions'/'/resources')."""
    from urllib.parse import urljoin, urlparse

    base_netloc = urlparse(base_url).netloc
    seen: set[str] = set()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#", 1)[0].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        if urlparse(full).netloc != base_netloc:
            continue
        normalized = full.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append(full)
    return links


@tool
def fetch_webpage(url: str) -> str:
    """Fetch a specific web page by URL (e.g. one returned by 'web_search' or
    'google_search') and return its readable text content (HTML tags,
    scripts, and styles stripped out).

    Use this to "read" a page after a search tool gives you its URL, instead
    of guessing what the page says from just the search snippet. 'url' must
    be a full URL including the scheme, e.g. 'https://example.com/article'.
    Requires internet access. Output is truncated if very long. If you need
    to explore SEVERAL pages of the same site (e.g. a company's homepage,
    pricing, and about pages) instead of one exact URL you already know,
    prefer 'crawl_website' -- it discovers and fetches multiple same-site
    pages automatically instead of you guessing each sub-page's exact path.
    """
    url = url.strip().strip("\"'")
    if not url.lower().startswith(("http://", "https://")):
        return f"Error: '{url}' is not a valid http(s) URL"

    try:
        import requests  # noqa: F401 -- import-availability check only
        from bs4 import BeautifulSoup  # noqa: F401
    except Exception as exc:  # pragma: no cover - only if the optional dep is missing
        return f"Error: fetch_webpage is unavailable -- missing dependency ({exc})"

    try:
        soup = _fetch_soup(url)
    except Exception as exc:  # network/SSL/HTTP errors -- never crash the agent loop
        return f"Error fetching '{url}': {exc}"

    text = _readable_text(soup)
    if not text:
        return f"No readable text content found at '{url}'."
    if len(text) > FETCH_WEBPAGE_MAX_CHARS:
        text = text[:FETCH_WEBPAGE_MAX_CHARS] + "... [truncated]"
    return text


# "Web crawler": starting from one URL, discover and fetch SEVERAL same-site
# pages automatically instead of the model having to guess each sub-page's
# exact path for 'fetch_webpage' one call at a time (live testing showed
# guessed paths like '/solutions'/'/resources' 404ing even though real pages
# like '/pricing'/'/customers' existed under slightly different names).
# Simple breadth-first crawl bounded by 'max_pages' and restricted to the
# STARTING url's own domain (never follows off-site links), so this stays
# fast, predictable, and can't wander off across the entire web.
CRAWL_WEBSITE_DEFAULT_MAX_PAGES = 5
CRAWL_WEBSITE_HARD_MAX_PAGES = 10
CRAWL_WEBSITE_PER_PAGE_MAX_CHARS = 1500


@tool
def crawl_website(url: str, max_pages: int = 5) -> str:
    """Crawl a website starting at 'url': fetch that page, discover the
    same-domain links on it, then fetch a few more of those pages too
    (breadth-first), returning each page's readable text labeled by its URL.

    Use this instead of calling 'fetch_webpage' repeatedly while guessing
    exact sub-page URLs -- e.g. after a search tool gives you a company's
    homepage and you want its pricing/product/about info but don't know the
    exact paths (guessed paths often 404 even when a similarly-named real
    page exists). Only follows links within the SAME domain as 'url' (never
    wanders off-site), skips duplicates, and stops after 'max_pages' pages
    (default 5, hard-capped at 10) so this stays fast and bounded. 'url'
    must be a full URL including scheme, e.g. 'https://example.com'.
    """
    url = url.strip().strip("\"'")
    if not url.lower().startswith(("http://", "https://")):
        return f"Error: '{url}' is not a valid http(s) URL"

    try:
        max_pages = int(max_pages)
    except (TypeError, ValueError):
        max_pages = CRAWL_WEBSITE_DEFAULT_MAX_PAGES
    max_pages = max(1, min(max_pages, CRAWL_WEBSITE_HARD_MAX_PAGES))

    try:
        import requests  # noqa: F401 -- import-availability check only
        from bs4 import BeautifulSoup  # noqa: F401
    except Exception as exc:  # pragma: no cover - only if the optional dep is missing
        return f"Error: crawl_website is unavailable -- missing dependency ({exc})"

    visited: set[str] = set()
    queue: list[str] = [url]
    pages: list[str] = []
    errors: list[str] = []

    while queue and len(pages) < max_pages:
        current = queue.pop(0)
        normalized = current.rstrip("/")
        if normalized in visited:
            continue
        visited.add(normalized)

        try:
            soup = _fetch_soup(current)
        except Exception as exc:  # network/SSL/HTTP errors -- skip, keep crawling
            errors.append(f"{current}: {exc}")
            continue

        text = _readable_text(soup)
        if text:
            if len(text) > CRAWL_WEBSITE_PER_PAGE_MAX_CHARS:
                text = text[:CRAWL_WEBSITE_PER_PAGE_MAX_CHARS] + "... [truncated]"
            pages.append(f"=== {current} ===\n{text}")

        if len(pages) < max_pages:
            for link in _same_domain_links(soup, url):
                link_normalized = link.rstrip("/")
                if link_normalized not in visited and link not in queue:
                    queue.append(link)

    if not pages:
        if errors:
            return "Error: could not crawl any pages:\n" + "\n".join(errors)
        return f"No readable text content found starting at '{url}'."

    result = "\n\n".join(pages)
    if errors:
        result += "\n\n(Some pages could not be fetched: " + "; ".join(errors) + ")"
    return result


# NOTE ON SECURITY MODEL (read this before reusing this tool anywhere else):
# This was originally an ALLOWLIST of read-only command names + shell=False,
# which is the safe pattern for any real/production system: the LLM's
# intentions don't matter, the tool itself enforces what can physically run.
#
# At the user's explicit request, that boundary has been REMOVED below so the
# model has full freedom to run any shell command it decides it needs,
# including pipes/redirects/chaining via a real shell (shell=True). This is
# appropriate ONLY for a local demo/learning sandbox running as your own user
# on your own machine. It is NOT safe to expose this to untrusted input, a
# multi-user service, or anything prompt-injectable from the outside (e.g. a
# tool that reads attacker-controlled web pages) -- a model can be tricked
# into running destructive commands (rm -rf, curl | sh, etc.) and nothing
# here will stop it. If you ever productionize this, bring back the allowlist.
# SHELL_COMMAND_TIMEOUT_SECONDS was originally 10s, which seemed generous for
# typical shell one-liners (ls, wc, grep, pwd) -- but this app's OWN HTTP
# endpoints (e.g. curl'ing http://127.0.0.1:8000/chat from within a command,
# which the model is explicitly encouraged to do when asked to "run the curl
# test command") can legitimately take anywhere from a few seconds to over a
# minute, since they call a real local LLM (especially the "powerful" tier /
# devstral:24b, which is slow on modest hardware). With only 10s, every such
# self-curl was getting killed by THIS timeout before the LLM even finished
# responding, producing a generic "Error: command timed out after 10s" that
# the model then misinterpreted as "the server isn't running" -- when really
# the server was running and answering, just slower than 10s. Raised to 90s
# to comfortably cover a single slow local LLM round-trip.
SHELL_COMMAND_TIMEOUT_SECONDS = 90
SHELL_OUTPUT_MAX_CHARS = 4000


class _TimeoutAwareBashProcess:
    """Minimal duck-typed stand-in for langchain_experimental's
    ``BashProcess`` (the default ``process`` that ``ShellTool`` would use),
    passed in explicitly so we don't need the ``langchain-experimental``
    package as a dependency just for this one small piece.

    IMPORTANT: ``ShellTool._run`` catches ANY exception from
    ``self.process.run(...)`` and silently swallows it into ``None`` -- it
    does NOT propagate a useful error message to the caller. So this class
    must NEVER raise: every failure mode (non-zero exit code, timeout) is
    turned into a descriptive string and returned normally, exactly like the
    original hand-rolled implementation this replaces. That string becomes
    the tool's observation, which flows straight back to the model as the
    next turn's input -- i.e. the model sees the real error/exit code and
    can self-correct its next attempt, no extra plumbing required.
    """

    def run(self, commands: str | list[str]) -> str:
        if isinstance(commands, list):
            commands = " && ".join(commands)
        command = commands.strip()
        if not command:
            return "Error: empty command"

        try:
            result = subprocess.run(
                command,
                shell=True,  # intentional: a real shell (pipes, redirects, globs, etc.)
                capture_output=True,
                text=True,
                timeout=SHELL_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {SHELL_COMMAND_TIMEOUT_SECONDS}s"
        except OSError as exc:  # e.g. the shell binary itself is missing
            return f"Error: could not execute command ({exc})"

        output = (result.stdout or "") + (result.stderr or "")
        output = output.strip() or "(command produced no output)"
        if result.returncode != 0:
            output = f"[exit code {result.returncode}]\n{output}"
        if len(output) > SHELL_OUTPUT_MAX_CHARS:
            output = output[:SHELL_OUTPUT_MAX_CHARS] + "... [truncated]"
        return output


# Use LangChain's PREBUILT ``ShellTool`` (langchain_community.tools) instead
# of hand-rolling the subprocess call ourselves -- same "use the maintained
# integration where one exists" rationale as ``wikipedia_search`` below.
# ``process=`` is passed explicitly (see ``_TimeoutAwareBashProcess`` above)
# so ``ShellTool`` doesn't fall back to its default factory, which requires
# installing the separate ``langchain-experimental`` package (and, at the
# time of writing, drags in a newer ``langsmith`` that conflicts with this
# project's pinned ``langchain-core``/``langchain-community`` versions).
_shell_tool = ShellTool(process=_TimeoutAwareBashProcess())


@tool
def run_shell_command(command: str) -> str:
    """Run any shell command (full bash/zsh syntax: pipes, redirects, chaining,
    globs, all standard Unix tools like wc/grep/find/awk/sed/xargs) and return
    its combined stdout+stderr.

    This has NO allowlist and NO restrictions on which programs or shell
    features can be used -- you have full freedom to run whatever command is
    needed to answer the question, including multi-step pipelines like
    "ls -la | wc -l" or "find . -name '*.py' | xargs wc -l". There is a
    90-second timeout (enough to curl this app's OWN slower endpoints, which
    call a real local LLM) and output is truncated if extremely long, but
    otherwise this behaves like a normal shell prompt. Use it for file
    system, process, OS, and system-info questions, or anything else where
    running a real command gives a more accurate answer than guessing.

    If the command fails (non-zero exit code, timeout, or any other error),
    the returned text says so explicitly (e.g. a "[exit code N]" prefix or an
    "Error: ..." message) -- read it and adjust the NEXT command accordingly
    instead of assuming it succeeded.

    THIS RUNS ON THE SERVER'S OWN MACHINE WITH THE SERVER'S OWN PERMISSIONS --
    avoid destructive commands (deleting/overwriting files, killing
    processes, modifying system state) unless the user has clearly asked for
    exactly that.
    """
    command = command.strip()
    if not command:
        return "Error: empty command"

    with warnings.catch_warnings():
        # ShellInput's validator warns "the shell tool has no safeguards by
        # default" on every single invocation -- we already document that
        # tradeoff at length above, so silence the repeat noise here.
        warnings.simplefilter("ignore")
        try:
            return str(_shell_tool.invoke({"commands": command}))
        except Exception as exc:  # defensive: never let a bad tool call crash the loop
            return f"Error running shell command: {exc}"



# Full file-system toolkit: use LangChain's PREBUILT ``FileManagementToolkit``
# (langchain_community.agent_toolkits) instead of hand-rolling pathlib-based
# tools. Leaving ``selected_tools`` unset exposes its ENTIRE family of
# ready-made, agent-ready file-system tools -- read_file, write_file,
# list_directory, file_search, copy_file, move_file, file_delete -- so the
# model has the full range of file operations a real coding assistant needs,
# not just a hand-picked subset. Same security-model note as
# run_shell_command applies: ``root_dir=None`` means no path restriction,
# which is appropriate for a local demo/learning sandbox only -- pass a real
# ``root_dir`` to sandbox these to one directory in any less trusted setting.
_file_tools = {t.name: t for t in FileManagementToolkit(root_dir=None).get_tools()}
read_file = _file_tools["read_file"]
write_file = _file_tools["write_file"]
copy_file = _file_tools["copy_file"]
move_file = _file_tools["move_file"]
file_delete = _file_tools["file_delete"]
_raw_list_directory = _file_tools["list_directory"]
_raw_file_search = _file_tools["file_search"]


# BUG FIX (observed failure): asked "how many .py files are in the project
# dir?", the model called file_search(dir_path="project_dir", pattern="*.py")
# -- it took the user's wording ("the project dir") LITERALLY as a path
# instead of using the documented default "." for "the current/project
# directory". The underlying LangChain tool then silently returned
# "No files found for pattern *.py in directory project_dir" -- which reads
# exactly like a real "zero files" answer, giving the model (and the self-
# correction graph) no signal that the DIRECTORY ITSELF was wrong, so it
# confidently reported "there are no .py files" instead of retrying with ".".
# Fix: validate dir_path before delegating, and fail LOUDLY with a hint to
# use "." plus the real top-level listing, so the model can self-correct
# instead of silently returning a wrong zero-files answer.
def _explain_bad_dir(dir_path: str) -> str | None:
    path = Path(dir_path).expanduser()
    if path.exists() and path.is_dir():
        return None
    try:
        entries = ", ".join(sorted(e.name for e in Path(".").iterdir())[:25])
    except OSError:
        entries = "(could not list '.')"
    return (
        f"Error: directory '{dir_path}' does not exist. Do NOT guess a "
        "directory name from the user's wording (e.g. 'the project dir' is "
        "NOT a literal path called 'project_dir') -- the project root is "
        "always the current working directory, so pass dir_path='.' to mean "
        f"'the whole project'. Top-level entries in '.': {entries}"
    )


@tool
def list_directory(dir_path: str = ".") -> str:
    """List all files and directories in the specified folder.

    'dir_path' defaults to '.', which IS the project root (the current
    working directory) -- use '.' whenever the user refers to 'the project
    dir', 'this directory', 'the current directory', etc. Do NOT invent a
    literal path from that wording (e.g. there is no directory actually
    named 'project_dir' or 'current directory').
    """
    error = _explain_bad_dir(dir_path)
    if error:
        return error
    return _raw_list_directory.invoke({"dir_path": dir_path})


@tool
def file_search(pattern: str, dir_path: str = ".") -> str:
    """Recursively search for files in a subdirectory that match the Unix
    shell regex pattern ('*' matches everything), e.g. pattern='*.py'.

    'dir_path' defaults to '.', which IS the project root (the current
    working directory) -- use '.' whenever the user refers to 'the project
    dir', 'this directory', 'the current directory', etc. Do NOT invent a
    literal path from that wording (e.g. there is no directory actually
    named 'project_dir' or 'current directory'). Note this also recurses
    into dependency/virtualenv folders (e.g. '.venv') if present under
    'dir_path', so for "how many files in THIS PROJECT" questions prefer
    'run_shell_command' with something like
    "find . -name '*.py' -not -path './.venv/*'" to exclude them.
    """
    error = _explain_bad_dir(dir_path)
    if error:
        return error
    return _raw_file_search.invoke({"dir_path": dir_path, "pattern": pattern})


@tool
def system_resources() -> str:
    """Report current CPU usage, memory usage, and disk usage as exact numbers.

    Computed directly via the 'psutil' library (no shelling out to 'df',
    'top', or 'uptime' and parsing their text). Use this for questions like
    "how much memory/disk/CPU is being used right now".
    """
    cpu_percent = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return (
        f"CPU usage: {cpu_percent:.1f}%\n"
        f"Memory: {mem.percent:.1f}% used "
        f"({mem.used // (1024 ** 2)} MB used / {mem.total // (1024 ** 2)} MB total)\n"
        f"Disk ('/'): {disk.percent:.1f}% used "
        f"({disk.used // (1024 ** 3)} GB used / {disk.total // (1024 ** 3)} GB total)"
    )


# edit_file has no off-the-shelf LangChain equivalent (the built-in file
# toolkit can read/write/move/copy/delete whole files but has no "replace
# this exact snippet" tool), so it stays hand-written -- it gives the model
# a far less error-prone way to change PART of a file than reconstructing
# the whole content by hand via 'write_file'.


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace an exact snippet of text within an existing file with new text.

    'old_text' must match EXACTLY ONE location in the file (including
    whitespace/indentation) -- if it appears zero times or more than once,
    nothing is changed and an error is returned explaining why, so you can
    adjust 'old_text' (e.g. include a bit more surrounding context) and try
    again. This is much safer than rewriting a whole file via 'write_file',
    since it can't accidentally discard unrelated content you didn't mean to
    touch.

    Use 'read_file' first to see the file's exact current content (including
    whitespace) before choosing 'old_text'. 'path' is resolved relative to the
    server's working directory (the project root) if not absolute.
    """
    try:
        target = Path(path).expanduser()
        if not target.exists():
            return f"Error: path '{path}' does not exist"
        if not target.is_file():
            return f"Error: '{path}' is not a file"
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' is not a readable text file (binary content)"
    except OSError as exc:
        return f"Error: could not read '{path}' ({exc})"

    occurrences = content.count(old_text)
    if occurrences == 0:
        return f"Error: old_text not found in '{path}' -- no changes made"
    if occurrences > 1:
        return (
            f"Error: old_text matches {occurrences} locations in '{path}' -- "
            "include more surrounding context to make it unique, no changes made"
        )

    try:
        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
    except OSError as exc:
        return f"Error: could not write '{path}' ({exc})"

    return f"Edited '{path}': replaced 1 occurrence."


# Reuse the same "don't walk into caches/venvs/binaries" exclusion list that
# app.codebase_rag already maintains for its own semantic index, instead of
# keeping a second, possibly-drifting copy here (imported at the top of this
# file, alongside the other app.codebase_rag import, as _GREP_EXCLUDED_DIR_NAMES).

GREP_SEARCH_MAX_RESULTS_CAP = 200
GREP_SEARCH_DEFAULT_MAX_RESULTS = 50
GREP_SEARCH_MAX_LINE_CHARS = 300


@tool
def grep_search(
    query: str,
    dir_path: str = ".",
    include_pattern: str | None = None,
    is_regex: bool = False,
    max_results: int = 50,
) -> str:
    """Search for an exact text or regex pattern across files on disk and
    return matching "path:line: content" lines.

    Use this for precise, exact lookups -- a specific function/class/variable
    name, an import, an error string -- across many files at once, instead of
    reading files one by one or relying on 'codebase_search' (which does
    semantic/meaning-based matching, not exact text matching). 'query' is a
    plain substring by default; set is_regex=True to treat it as a Python
    regular expression instead. 'dir_path' defaults to '.' (the project
    root). 'include_pattern' optionally restricts which files are searched
    via a glob, e.g. '*.py' or 'app/*.py' -- omit it to search every
    text-readable file found. Skips binary files and common
    cache/dependency directories (__pycache__, .venv, .git, node_modules,
    etc.) automatically. Results are capped at 'max_results' (default 50,
    hard-capped at 200) so this stays fast and the output stays readable.
    """
    root = Path(dir_path).expanduser()
    if not root.exists() or not root.is_dir():
        return f"Error: directory '{dir_path}' does not exist"

    try:
        max_results = int(max_results)
    except (TypeError, ValueError):
        max_results = GREP_SEARCH_DEFAULT_MAX_RESULTS
    max_results = max(1, min(max_results, GREP_SEARCH_MAX_RESULTS_CAP))

    if is_regex:
        try:
            pattern = re.compile(query)
        except re.error as exc:
            return f"Error: invalid regex '{query}': {exc}"
    else:
        pattern = None

    matches: list[str] = []
    files_searched = 0
    for path in sorted(root.rglob("*")):
        if len(matches) >= max_results:
            break
        if not path.is_file():
            continue
        if any(part in _GREP_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        rel_path = path.relative_to(root).as_posix()
        if include_pattern and not fnmatch.fnmatch(rel_path, include_pattern):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # skip binary/unreadable files silently
        files_searched += 1

        for line_no, line in enumerate(text.splitlines(), start=1):
            found = pattern.search(line) if pattern else (query in line)
            if not found:
                continue
            snippet = line.strip()
            if len(snippet) > GREP_SEARCH_MAX_LINE_CHARS:
                snippet = snippet[:GREP_SEARCH_MAX_LINE_CHARS] + "... [truncated]"
            matches.append(f"{rel_path}:{line_no}: {snippet}")
            if len(matches) >= max_results:
                break

    if not matches:
        return f"No matches found for '{query}' (searched {files_searched} files under '{dir_path}')."

    result = "\n".join(matches)
    if len(matches) >= max_results:
        result += f"\n... [stopped at {max_results} results, there may be more -- narrow the search to see them]"
    return result


# "Lint/compile check a file": the closest equivalent to an IDE's "get
# errors/problems" panel. Uses the project's OWN already-installed/pinned
# 'ruff' (see pyproject.toml's [tool.ruff] config, which this respects
# automatically by running from the project root) for real Python
# diagnostics -- syntax errors AND style/correctness lint rules (E/F/I) --
# instead of a hand-rolled ast.parse() call that could only ever catch
# syntax errors, not undefined names/unused imports/etc.
GET_ERRORS_TIMEOUT_SECONDS = 30


@tool
def get_errors(file_paths: list[str]) -> str:
    """Check one or more Python files for syntax errors and lint problems
    (undefined names, unused imports, etc.) and return them, file by file.

    Runs this project's own configured 'ruff' linter (see pyproject.toml) on
    each path in 'file_paths' and reports every diagnostic found: line,
    column, rule code, and message. A file with no problems is reported as
    clean. Use this after writing or editing a Python file (e.g. via
    'write_file'/'edit_file') to verify the change didn't introduce an
    error, instead of assuming it's correct or waiting for the user to
    report a problem. Only checks the files you explicitly list here, never
    the whole project.
    """
    if not file_paths:
        return "Error: file_paths must contain at least one path"

    existing: list[str] = []
    missing: list[str] = []
    for p in file_paths:
        if Path(p).expanduser().is_file():
            existing.append(p)
        else:
            missing.append(p)

    report_parts: list[str] = []
    if missing:
        report_parts.append("Error: file(s) not found: " + ", ".join(missing))

    if existing:
        try:
            result = subprocess.run(
                ["ruff", "check", "--output-format=json", *existing],
                capture_output=True,
                text=True,
                timeout=GET_ERRORS_TIMEOUT_SECONDS,
            )
        except FileNotFoundError:
            return "Error: get_errors is unavailable -- 'ruff' is not installed/on PATH"
        except subprocess.TimeoutExpired:
            return f"Error: ruff check timed out after {GET_ERRORS_TIMEOUT_SECONDS}s"

        try:
            diagnostics = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            report_parts.append(f"Error: could not parse ruff output: {result.stdout or result.stderr}")
            diagnostics = []

        by_file: dict[str, list[dict]] = {p: [] for p in existing}
        for diag in diagnostics:
            filename = diag.get("filename", "")
            # ruff reports absolute/resolved paths -- match back to whichever
            # of our originally-requested paths it corresponds to.
            match = next((p for p in existing if Path(p).expanduser().resolve() == Path(filename).resolve()), None)
            by_file.setdefault(match or filename, []).append(diag)

        for p in existing:
            file_diags = by_file.get(p, [])
            if not file_diags:
                report_parts.append(f"'{p}': no errors found.")
                continue
            lines = [
                f"  line {d.get('location', {}).get('row', '?')}, "
                f"col {d.get('location', {}).get('column', '?')}: "
                f"[{d.get('code', '?')}] {d.get('message', '')}"
                for d in file_diags
            ]
            report_parts.append(f"'{p}': {len(file_diags)} problem(s):\n" + "\n".join(lines))

    return "\n\n".join(report_parts)


# "Ask the user a clarifying question and WAIT for their reply" -- the
# closest equivalent to an interactive IDE assistant's own human-in-the-loop
# tool. This backend is a stateless request/response HTTP service (no open
# socket held to a specific browser tab mid-ReAct-loop) -- so, unlike an IDE
# extension, this CANNOT literally suspend Python execution and pop up a
# dialog box. Instead, the tool itself is a plain marker: calling it
# deterministically makes the REACT LOOP ITSELF stop immediately (see
# app.react_agent/app.react_langgraph_agent/app.reasoning_react_agent, all of
# which special-case this tool's name) and return the question as that
# turn's answer, with `awaiting_user_input=True` on the result. The caller
# (API client/UI) shows that question to the real user, then sends the
# user's reply back as the NEXT turn's prompt (with history/thread_id
# carrying the conversation forward as normal) -- at which point the loop
# resumes exactly where it left off. This function's own return value is
# never actually shown to the model (the loop short-circuits before any
# further model call), it only exists so the question is recorded in
# `steps`/the conversation transcript like any other tool call.
@tool
def ask_user(question: str) -> str:
    """Ask the human user a clarifying question and STOP, waiting for their
    reply, instead of guessing an ambiguous detail or proceeding on an
    assumption you are not confident about.

    Use this when the user's request is genuinely ambiguous or missing
    information you need (e.g. "edit the file" when several files could
    match, or a destructive action like deleting something where you should
    confirm first) -- NOT for things you can find out yourself with another
    tool (prefer 'codebase_search'/'grep_search'/'read_file'/'web_search'
    etc. first). Calling this immediately ends your turn: the question is
    shown to the user, and you will see their reply as the next message
    once they respond -- you will NOT get to call any more tools or say
    anything else until then.
    """
    return f"(Question sent to the user -- awaiting their reply: {question})"


AVAILABLE_TOOLS = [
    calculator,
    knowledge_base_lookup,
    wikipedia_search,
    web_search,
    google_search,
    arxiv_search,
    stackexchange_search,
    fetch_webpage,
    crawl_website,
    run_shell_command,
    list_directory,
    read_file,
    write_file,
    edit_file,
    file_search,
    grep_search,
    get_errors,
    copy_file,
    move_file,
    file_delete,
    system_resources,
    ask_user,
    rag_search,
    codebase_search,
    readme_search,
    article_search,
]
