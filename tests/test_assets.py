"""Tests for paper-bundle asset materialization."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from arxiv2md.assets import AssetLimits, AssetMaterializer

PNG = b"\x89PNG\r\n\x1a\ncontent"


@pytest.mark.asyncio
async def test_materializer_downloads_deterministically_and_writes_manifest(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start.png":
            return httpx.Response(302, headers={"location": "/final.png"})
        return httpx.Response(200, headers={"content-type": "image/png"}, content=PNG)

    output = tmp_path / "paper.md"
    materializer = AssetMaterializer(output, transport=httpx.MockTransport(handler))
    source = "https://arxiv.org/start.png"
    await materializer.materialize([source, source])

    digest = hashlib.sha256(PNG).hexdigest()
    reference = f"paper_assets/figure-001-{digest[:12]}.png"
    assert materializer(source) == reference
    assert (tmp_path / reference).read_bytes() == PNG
    manifest = json.loads((tmp_path / "paper_assets/manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["converter"] == "arxiv2markdown"
    assert manifest["assets"][0]["sha256"] == digest
    assert manifest["assets"][0]["resolved_source"] == "https://arxiv.org/final.png"


@pytest.mark.asyncio
async def test_materializer_revalidates_redirect_origin(tmp_path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": "https://example.com/image.png"}))
    materializer = AssetMaterializer(tmp_path / "paper.md", transport=transport)
    with pytest.raises(ValueError, match="Untrusted asset URL"):
        await materializer.materialize(["https://arxiv.org/image.png"])


@pytest.mark.asyncio
async def test_materializer_enforces_signature_and_limits(tmp_path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "image/png"}, content=b"not png"))
    materializer = AssetMaterializer(tmp_path / "paper.md", limits=AssetLimits(max_count=1), transport=transport)
    with pytest.raises(ValueError, match="signature"):
        await materializer.materialize(["https://arxiv.org/image.png"])
    with pytest.raises(ValueError, match="count"):
        await materializer.materialize(["https://arxiv.org/a.png", "https://arxiv.org/b.png"])
