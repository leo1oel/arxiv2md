"""Tests for fetch-time validation of what arXiv actually returned."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import arxiv2md.fetch as fetch_module
from arxiv2md.fetch import fetch_arxiv_html

PAPER_HTML = '<html><body><article class="ltx_document"><p>Body.</p></article></body></html>'
ABS_PAGE_HTML = "<html><body><h1>Title: Some paper</h1><p>Subjects: cs.LG</p></body></html>"


def _install_transport(monkeypatch, tmp_path, handler):
    """Route arxiv2md.fetch's HTTP through a MockTransport, cache under tmp."""

    def async_client(**kwargs):
        kwargs.pop("transport", None)
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(
        fetch_module,
        "httpx",
        SimpleNamespace(
            AsyncClient=async_client,
            Timeout=httpx.Timeout,
            RequestError=httpx.RequestError,
            HTTPStatusError=httpx.HTTPStatusError,
        ),
    )
    monkeypatch.setattr(fetch_module, "ARXIV2MD_CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(fetch_module, "ARXIV2MD_FETCH_MAX_RETRIES", 0)
    monkeypatch.setattr(fetch_module, "ARXIV2MD_FETCH_BACKOFF_S", 0.0)


@pytest.mark.asyncio
async def test_a_non_rendering_page_falls_back_to_ar5iv(monkeypatch, tmp_path) -> None:
    # ar5iv answers 200 for papers it cannot render by redirecting to the
    # abstract page; before validation that page became a stub "paper".
    def handler(request: httpx.Request) -> httpx.Response:
        body = PAPER_HTML if "ar5iv" in request.url.host else ABS_PAGE_HTML
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body.encode())

    _install_transport(monkeypatch, tmp_path, handler)
    html, source_url = await fetch_arxiv_html(
        "https://arxiv.org/html/9901002",
        arxiv_id="cs/9901002",
        version=None,
        use_cache=False,
        ar5iv_url="https://ar5iv.labs.arxiv.org/html/cs/9901002",
    )
    assert "ltx_document" in html
    assert "ar5iv" in source_url


@pytest.mark.asyncio
async def test_no_rendering_anywhere_is_an_error_not_a_stub(monkeypatch, tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=ABS_PAGE_HTML.encode())

    _install_transport(monkeypatch, tmp_path, handler)
    with pytest.raises(RuntimeError, match="does not have an HTML version"):
        await fetch_arxiv_html(
            "https://arxiv.org/html/9901002",
            arxiv_id="cs/9901002",
            version=None,
            use_cache=False,
            ar5iv_url="https://ar5iv.labs.arxiv.org/html/cs/9901002",
        )


@pytest.mark.asyncio
async def test_a_cached_non_rendering_does_not_satisfy_the_request(monkeypatch, tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=PAPER_HTML.encode())

    _install_transport(monkeypatch, tmp_path, handler)
    cache_dir = tmp_path / "cache" / "cs_9901002__latest"
    cache_dir.mkdir(parents=True)
    (cache_dir / "source.html").write_text(ABS_PAGE_HTML, encoding="utf-8")

    html, _ = await fetch_arxiv_html(
        "https://arxiv.org/html/9901002",
        arxiv_id="cs/9901002",
        version=None,
        use_cache=True,
    )
    assert "ltx_document" in html
    # The refetched rendering replaced the cached stub.
    assert "ltx_document" in (cache_dir / "source.html").read_text(encoding="utf-8")
