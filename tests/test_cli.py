"""CLI contract tests."""

from __future__ import annotations

from argparse import Namespace

import pytest

from arxiv2md.__main__ import _async_main


@pytest.mark.asyncio
async def test_download_assets_rejects_stdout_before_fetch() -> None:
    args = Namespace(
        input_text="2501.11120v1",
        output="-",
        download_assets=True,
        sections=None,
        section=None,
        remove_refs=False,
        remove_toc=False,
        remove_inline_citations=False,
        section_filter_mode="exclude",
        frontmatter=False,
        include_tree=False,
    )
    with pytest.raises(ValueError, match="stdout is not supported"):
        await _async_main(args)
