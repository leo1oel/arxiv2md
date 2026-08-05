"""Ingestion pipeline for arXiv HTML -> Markdown."""

from __future__ import annotations

from collections.abc import Callable

from bs4 import BeautifulSoup

from arxiv2md.assets import AssetMaterializer, resolve_asset_url
from arxiv2md.fetch import fetch_arxiv_html
from arxiv2md.html_parser import parse_arxiv_html
from arxiv2md.markdown import convert_fragment_to_markdown
from arxiv2md.output_formatter import format_paper
from arxiv2md.schemas import IngestionResult
from arxiv2md.sections import filter_sections

_REFERENCE_TITLES = ("references", "bibliography")
_ABSTRACT_TITLE = "abstract"


async def ingest_paper(
    *,
    arxiv_id: str,
    version: str | None,
    html_url: str,
    ar5iv_url: str | None = None,
    remove_refs: bool,
    remove_toc: bool,
    remove_inline_citations: bool = False,
    section_filter_mode: str,
    sections: list[str],
    include_frontmatter: bool = False,
    asset_materializer: AssetMaterializer | None = None,
) -> tuple[IngestionResult, dict[str, str | list[str] | None]]:
    """Fetch, parse, and serialize an arXiv paper into Markdown.

    Parameters
    ----------
    remove_inline_citations : bool
        If True, completely remove inline citation links from the output.
        If False (default), citation URLs are stripped but text is kept.
    """
    html, source_url = await fetch_arxiv_html(html_url, arxiv_id=arxiv_id, version=version, use_cache=True, ar5iv_url=ar5iv_url)
    parsed = parse_arxiv_html(html)

    filtered_sections = filter_sections(parsed.sections, mode=section_filter_mode, selected=sections)
    if remove_refs:
        filtered_sections = filter_sections(filtered_sections, mode="exclude", selected=_REFERENCE_TITLES)

    # Check if abstract should be included based on section filter
    selected_lower = [s.lower() for s in sections]
    if section_filter_mode == "exclude":
        include_abstract = _ABSTRACT_TITLE not in selected_lower
    else:  # include mode
        include_abstract = not sections or _ABSTRACT_TITLE in selected_lower

    if asset_materializer:
        asset_urls: list[str] = []
        for section in filtered_sections:
            _collect_asset_urls(section, source_url, asset_urls)
        await asset_materializer.materialize(asset_urls)

    for section in filtered_sections:
        _populate_section_markdown(section, remove_inline_citations=remove_inline_citations, base_url=source_url, asset_materializer=asset_materializer)

    result = format_paper(
        arxiv_id=arxiv_id,
        version=version,
        title=parsed.title,
        authors=parsed.authors,
        abstract=parsed.abstract if include_abstract else None,
        sections=filtered_sections,
        include_toc=not remove_toc,
        include_abstract_in_tree=parsed.abstract is not None,
        include_frontmatter=include_frontmatter,
    )

    metadata = {
        "title": parsed.title,
        "authors": parsed.authors,
        "abstract": parsed.abstract,
    }

    return result, metadata


def _populate_section_markdown(section, *, remove_inline_citations: bool = False, base_url: str | None = None, asset_materializer: Callable[[str], str] | None = None) -> None:
    if section.html:
        section.markdown = convert_fragment_to_markdown(section.html, remove_inline_citations=remove_inline_citations, base_url=base_url, asset_materializer=asset_materializer)
    for child in section.children:
        _populate_section_markdown(child, remove_inline_citations=remove_inline_citations, base_url=base_url, asset_materializer=asset_materializer)


def _collect_asset_urls(section, base_url: str, urls: list[str]) -> None:
    if section.html:
        soup = BeautifulSoup(section.html, "html.parser")
        urls.extend(resolve_asset_url(base_url, str(image["src"])) for image in soup.find_all("img", src=True))
    for child in section.children:
        _collect_asset_urls(child, base_url, urls)
