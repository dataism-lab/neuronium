"""Local web tools for article review demos (T0.2)."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from neuronium_agent.tools.local_tools import ToolExecutionError


def invoke_web_fetch_html(
    args: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    del policy, runtime
    raw_url = str(args.get("url", "")).strip()
    if not raw_url:
        raise ToolExecutionError("web.fetch_html: missing 'url'")

    timeout_seconds = int(args.get("timeout_seconds", 20))
    max_bytes = int(args.get("max_bytes", 2_000_000))
    if timeout_seconds <= 0:
        raise ToolExecutionError("web.fetch_html: timeout_seconds must be > 0")
    if max_bytes <= 0:
        raise ToolExecutionError("web.fetch_html: max_bytes must be > 0")

    req = Request(
        raw_url,
        headers={
            "User-Agent": (
                "NeuroniumAgent/0.1 (+https://github.com/neuronium-agent)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        },
        method="GET",
    )
    warnings: list[str] = []
    try:
        with urlopen(req, timeout=timeout_seconds) as resp:
            content_type = str(resp.headers.get("Content-Type", ""))
            data = resp.read(max_bytes + 1)
            truncated = len(data) > max_bytes
            if truncated:
                warnings.append(f"response_truncated_to_{max_bytes}_bytes")
                data = data[:max_bytes]

            charset = resp.headers.get_content_charset() or "utf-8"
            html = data.decode(charset, errors="replace")
            return {
                "final_url": str(resp.geturl()),
                "status_code": int(getattr(resp, "status", 200)),
                "html": html,
                "content_type": content_type,
                "warnings": warnings,
            }
    except HTTPError as exc:
        raise ToolExecutionError(
            f"web.fetch_html: HTTP {exc.code} for {raw_url}"
        ) from exc
    except URLError as exc:
        raise ToolExecutionError(f"web.fetch_html: network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ToolExecutionError("web.fetch_html: timeout") from exc
    except Exception as exc:
        raise ToolExecutionError(f"web.fetch_html: unexpected error: {exc}") from exc


def invoke_web_extract_article(
    args: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
    runtime: Any | None = None,
) -> dict[str, Any]:
    del policy, runtime
    request_url = str(args.get("url", "")).strip()
    final_url = str(args.get("final_url", "")).strip()
    base_url = final_url or request_url
    html = str(args.get("html", ""))
    if not base_url:
        raise ToolExecutionError("web.extract_article: missing 'url'")
    if not html:
        raise ToolExecutionError("web.extract_article: missing 'html'")

    try:
        from bs4 import BeautifulSoup  # type: ignore[import-untyped]
    except Exception:
        parser = _FallbackArticleParser(
            _resolve_effective_base_url(base_url=base_url, html=html)
        )
        parser.feed(html)
        parser.close()
        return parser.to_payload()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    base_url = _resolve_effective_base_url(base_url=base_url, html=html, soup=soup)

    title_guess = _extract_title_guess_bs4(soup)
    article_root = soup.find("article") or soup.find("main") or soup.body or soup
    text = article_root.get_text(separator="\n", strip=True)
    text = "\n".join(line for line in text.splitlines() if line.strip())

    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for img in article_root.find_all("img"):
        src = str(img.get("src", "")).strip()
        if not src or src.startswith("data:"):
            continue
        src_abs = urljoin(base_url, src)
        if src_abs in seen:
            continue
        seen.add(src_abs)
        alt = unescape(str(img.get("alt", "")).strip())
        images.append({"src": src_abs, "alt": alt})

    return {"title_guess": title_guess, "text": text, "images": images}


def _extract_title_guess_bs4(soup: Any) -> str:
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return str(og_title["content"]).strip()

    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
    if tw_title and tw_title.get("content"):
        return str(tw_title["content"]).strip()

    if soup.title and soup.title.string:
        return str(soup.title.string).strip()
    return ""


def _resolve_effective_base_url(*, base_url: str, html: str, soup: Any | None = None) -> str:
    """Resolve base URL used to expand relative links/images."""
    candidate = base_url.strip()
    if soup is not None:
        try:
            base_tag = soup.find("base")
            href = str(base_tag.get("href", "")).strip() if base_tag else ""
            if href:
                candidate = urljoin(candidate, href)
        except Exception:
            pass
    else:
        # Fallback mode when BeautifulSoup is unavailable.
        match = re.search(
            r"<base[^>]+href=['\"]([^'\"]+)['\"]",
            html,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = urljoin(candidate, match.group(1).strip())
    return _normalize_base_url_for_relative(candidate)


def _normalize_base_url_for_relative(url: str) -> str:
    """Treat extension-less terminal segment as directory-like URL."""
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    path = parts.path or "/"
    if path.endswith("/"):
        return url
    last_segment = path.rsplit("/", 1)[-1]
    # Heuristic for pages like /html/2511.12869v2 where relative assets are siblings.
    if "." not in last_segment:
        path = path + "/"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return url


class _FallbackArticleParser(HTMLParser):
    """Stdlib fallback parser when BeautifulSoup is unavailable."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._images: list[dict[str, str]] = []
        self._images_seen: set[str] = set()
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {k.lower(): (v or "") for k, v in attrs}
        t = tag.lower()
        if t in {"script", "style", "noscript", "template"}:
            self._skip_depth += 1
            return

        if t == "title":
            self._in_title = True
            return

        if self._skip_depth > 0:
            return

        if t == "img":
            src = attrs_map.get("src", "").strip()
            if not src or src.startswith("data:"):
                return
            src_abs = urljoin(self._base_url, src)
            if src_abs in self._images_seen:
                return
            self._images_seen.add(src_abs)
            self._images.append({
                "src": src_abs,
                "alt": unescape(attrs_map.get("alt", "").strip()),
            })
            return

        if t in {"p", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "template"} and self._skip_depth > 0:
            self._skip_depth -= 1
        if t == "title":
            self._in_title = False
        if t in {"p", "li", "article", "main", "section", "div"}:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not data.strip() or self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data.strip())
            return
        self._text_parts.append(data.strip())

    def to_payload(self) -> dict[str, Any]:
        title_guess = " ".join(x for x in self._title_parts if x).strip()
        text = "\n".join(line for line in "".join(self._text_parts).splitlines() if line.strip())
        return {"title_guess": title_guess, "text": text, "images": self._images}
