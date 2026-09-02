"""Tests for the ingestion pipeline's asset collection."""

from __future__ import annotations

import json

from arxiv2md.assets import AssetMaterializer
from arxiv2md.ingestion import (
    _collect_asset_urls,
    _materialize_inline_svg_assets,
    _populate_section_markdown,
)
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


async def test_inline_latexml_svg_survives_inside_its_figure_panel(tmp_path) -> None:
    section = _section(
        """
        <figure id="S4.F6"><div class="ltx_flex_figure">
          <div class="ltx_flex_cell ltx_flex_size_2">
            <figure id="S4.F6.sf1" class="ltx_figure_panel">
              <svg id="S4.F6.sf1.pic1" class="ltx_picture" viewBox="0 0 10 10">
                <clipPath id="plot"><path d="M0 0h10v10z"></path></clipPath>
                <foreignObject width="10" height="10"><span>Accuracy <math><mi>x</mi></math></span></foreignObject>
              </svg>
              <figcaption>(a) Pruning analysis</figcaption>
            </figure>
          </div>
        </div></figure>
        """
    )
    materializer = AssetMaterializer(tmp_path / "paper.md", compress=True)

    _materialize_inline_svg_assets(section, materializer)
    await materializer.materialize([])
    _populate_section_markdown(
        section,
        base_url="https://arxiv.org/html/2408.05088v1/",
        asset_materializer=materializer,
    )

    assert '<PaperFigurePanel id="S4.F6.sf1">' in section.markdown
    assert "![Figure](paper_assets/figure-inline-001-" in section.markdown
    assert "*(a) Pruning analysis*" in section.markdown
    manifest = json.loads((tmp_path / "paper_assets/manifest.json").read_text())
    assert manifest["assets"][0]["type"] == "image/svg+xml"
    svg = (tmp_path / manifest["assets"][0]["path"]).read_text()
    assert 'viewBox="0 0 10 10"' in svg
    assert "<clipPath" in svg
    assert "<foreignObject" in svg
    assert 'xmlns="http://www.w3.org/1999/xhtml"' in svg
    assert 'xmlns="http://www.w3.org/1998/Math/MathML"' in svg
