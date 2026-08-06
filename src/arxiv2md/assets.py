"""Safe, deterministic materialization of remote paper images."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

CONVERTER = "arxiv2markdown"
MANIFEST_SCHEMA = 1
TRUSTED_HOSTS = {"arxiv.org", "www.arxiv.org", "ar5iv.labs.arxiv.org"}
MIME_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}


def sniff_image_type(data: bytes) -> str | None:
    """The media type a payload actually is, or None if it is not an image.

    The bytes decide, not the Content-Type header. arXiv derives that header
    from the file name, so a figure an author saved as PNG and named .jpg
    arrives announced as image/jpeg — trusting the announcement rejected a
    perfectly good image and failed the whole paper. Sniffing is also the
    stricter half of the check: it is what proves the payload is an image at
    all, whatever the server claims.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def document_base_url(response_url: str, base_href: str | None) -> str:
    """The URL relative references in a document resolve against.

    A document's own ``<base href>`` wins over the URL it was fetched from,
    exactly as it does in a browser. arXiv relies on this: it answers
    ``/html/<id>`` directly and declares ``<base href="/html/<id>v<n>/">``, so
    resolving against the request URL instead drops both the version and the
    directory and sends every relative figure to ``/html/<file>``.
    """
    return urljoin(response_url, base_href) if base_href else response_url


def resolve_asset_url(source_url: str, src: str) -> str:
    """Resolve an image URL against the document's base URL."""
    return urljoin(source_url, src)


@dataclass(frozen=True)
class AssetLimits:
    # The byte budgets below are the real resource guard; this count only stops
    # a pathological page from opening thousands of connections. A survey paper
    # rendered by ar5iv legitimately carries a couple of hundred figures — 100
    # rejected Flamingo (235) outright rather than dropping a single image.
    max_count: int = 500
    max_file_bytes: int = 20 * 1024 * 1024
    max_total_bytes: int = 100 * 1024 * 1024
    timeout_seconds: float = 15.0
    max_redirects: int = 5


class AssetMaterializer:
    """Download validated raster images and map source URLs to local paths."""

    def __init__(self, output_file: Path, *, limits: AssetLimits = AssetLimits(), transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.output_file = output_file.resolve()
        self.assets_dir = self.output_file.parent / f"{self.output_file.stem}_assets"
        self.limits = limits
        self.transport = transport
        self._references: dict[str, str] = {}

    def __call__(self, source_url: str) -> str:
        return self._references.get(source_url, source_url)

    async def materialize(self, source_urls: list[str]) -> None:
        unique = list(dict.fromkeys(source_urls))
        if len(unique) > self.limits.max_count:
            raise ValueError(f"Asset count exceeds limit ({self.limits.max_count})")
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        root = self.assets_dir.resolve()
        entries: list[dict[str, object]] = []
        total = 0
        timeout = httpx.Timeout(self.limits.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=self.transport) as client:
            for index, source in enumerate(unique, 1):
                response, final_url = await self._get_following_safe_redirects(client, source)
                data = response.content
                if len(data) > self.limits.max_file_bytes:
                    raise ValueError(f"Asset exceeds per-file limit: {source}")
                total += len(data)
                if total > self.limits.max_total_bytes:
                    raise ValueError("Assets exceed total download limit")
                content_type = sniff_image_type(data)
                if content_type is None:
                    declared = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    raise ValueError(f"Asset signature does not match a supported image (declared {declared or 'nothing'}): {source}")
                extension = MIME_EXTENSIONS[content_type]
                digest = hashlib.sha256(data).hexdigest()
                name = f"figure-{index:03d}-{digest[:12]}{extension}"
                destination = (root / name).resolve()
                if root not in destination.parents:
                    raise ValueError("Asset path escapes output directory")
                destination.write_bytes(data)
                relative = destination.relative_to(self.output_file.parent).as_posix()
                self._references[source] = relative
                entries.append({"source": source, "resolved_source": final_url, "path": relative, "type": content_type, "size": len(data), "sha256": digest})
        manifest = {"schema_version": MANIFEST_SCHEMA, "converter": CONVERTER, "converter_version": "0.1.0", "assets": entries}
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    async def _get_following_safe_redirects(self, client: httpx.AsyncClient, url: str) -> tuple[httpx.Response, str]:
        current = url
        for _ in range(self.limits.max_redirects + 1):
            _validate_url(current)
            response = await client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Asset redirect has no location")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            return response, str(response.url)
        raise ValueError("Too many asset redirects")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in TRUSTED_HOSTS or parsed.username or parsed.password:
        raise ValueError(f"Untrusted asset URL: {url}")


