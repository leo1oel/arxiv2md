"""Tests for paper-bundle asset materialization."""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from arxiv2md.assets import AssetLimits, AssetMaterializer, document_base_url, resolve_asset_url, sniff_image_type

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


def test_document_base_url_honours_a_declared_base() -> None:
    # arXiv answers /html/<id> directly and declares the versioned directory,
    # so a relative figure has to land under it.
    base = document_base_url("https://arxiv.org/html/2106.09685", "/html/2106.09685v2/")
    assert base == "https://arxiv.org/html/2106.09685v2/"
    assert resolve_asset_url(base, "x1.png") == "https://arxiv.org/html/2106.09685v2/x1.png"


def test_document_base_url_falls_back_to_the_response_url() -> None:
    base = document_base_url("https://ar5iv.labs.arxiv.org/html/2106.09685", None)
    assert base == "https://ar5iv.labs.arxiv.org/html/2106.09685"
    # ar5iv uses root-relative sources, which resolve without a <base>.
    assert resolve_asset_url(base, "/html/2106.09685/assets/x1.png") == "https://ar5iv.labs.arxiv.org/html/2106.09685/assets/x1.png"


def test_document_base_url_accepts_an_absolute_base() -> None:
    base = document_base_url("https://arxiv.org/html/2106.09685", "https://cdn.example.com/paper/")
    assert base == "https://cdn.example.com/paper/"


@pytest.mark.asyncio
async def test_materializer_trusts_the_signature_over_a_mislabelled_content_type(tmp_path) -> None:
    # arXiv names the type after the file name, so a PNG an author saved as
    # .jpg arrives announced as image/jpeg. The bytes are what count.
    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "image/jpeg"}, content=PNG))
    materializer = AssetMaterializer(tmp_path / "paper.md", transport=transport)
    source = "https://arxiv.org/html/2410.08202v3/chinese.jpg"
    await materializer.materialize([source])

    assert materializer(source).endswith(".png")
    manifest = json.loads((tmp_path / "paper_assets/manifest.json").read_text())
    assert manifest["assets"][0]["type"] == "image/png"


def test_sniff_image_type_recognizes_each_supported_format() -> None:
    assert sniff_image_type(PNG) == "image/png"
    assert sniff_image_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_image_type(b"GIF89a rest") == "image/gif"
    assert sniff_image_type(b"RIFF\x00\x00\x00\x00WEBPrest") == "image/webp"
    assert sniff_image_type(b"<html>not an image</html>") is None
