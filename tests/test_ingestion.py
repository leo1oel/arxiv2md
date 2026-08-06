"""Tests for the ingestion pipeline's asset collection."""

from __future__ import annotations

from arxiv2md.ingestion import _collect_asset_urls
from arxiv2md.schemas import SectionNode


def _section(html: str, children: list[SectionNode] | None = None) -> SectionNode:
    return SectionNode(title="S", level=1, html=html, children=children or [])


def test_collect_asset_urls_resolves_against_the_document_base() -> None:
    section = _section('<p><img src="x1.png"/></p>', [_section('<p><img src="extracted/1/fig.jpg"/></p>')])
    urls: list[str] = []
    _collect_asset_urls(section, "https://arxiv.org/html/2106.09685v2/", urls)
    assert urls == [
        "https://arxiv.org/html/2106.09685v2/x1.png",
        "https://arxiv.org/html/2106.09685v2/extracted/1/fig.jpg",
    ]


def test_collect_asset_urls_skips_inline_data_images() -> None:
    section = _section('<p><img src="data:image/png;base64,iVBORw0KGgo="/><img src="x1.png"/></p>')
    urls: list[str] = []
    _collect_asset_urls(section, "https://arxiv.org/html/2106.09685v2/", urls)
    assert urls == ["https://arxiv.org/html/2106.09685v2/x1.png"]
