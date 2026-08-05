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


def resolve_asset_url(source_url: str, src: str) -> str:
    """Resolve an image URL against the actual HTML response URL."""
    return urljoin(source_url, src)


@dataclass(frozen=True)
class AssetLimits:
    max_count: int = 100
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
                content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                extension = MIME_EXTENSIONS.get(content_type)
                if extension is None:
                    raise ValueError(f"Unsupported asset content type: {content_type or 'missing'}")
                data = response.content
                if len(data) > self.limits.max_file_bytes:
                    raise ValueError(f"Asset exceeds per-file limit: {source}")
                total += len(data)
                if total > self.limits.max_total_bytes:
                    raise ValueError("Assets exceed total download limit")
                _validate_signature(content_type, data)
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


def _validate_signature(content_type: str, data: bytes) -> None:
    valid = {
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": data.startswith(b"\xff\xd8\xff"),
        "image/gif": data.startswith((b"GIF87a", b"GIF89a")),
        "image/webp": len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }[content_type]
    if not valid:
        raise ValueError(f"Asset signature does not match {content_type}")
