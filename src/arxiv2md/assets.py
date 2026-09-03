"""Safe, deterministic materialization of remote paper images."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from arxiv2md.utils.logging_config import get_logger

logger = get_logger(__name__)

CONVERTER = "arxiv2markdown"
MANIFEST_SCHEMA = 1
TRUSTED_HOSTS = {"arxiv.org", "www.arxiv.org", "ar5iv.labs.arxiv.org"}
MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
}
_SVG_ROOT = re.compile(
    rb"\A(?:\xef\xbb\xbf)?\s*(?:<\?xml\b.*?\?>\s*)?(?:<!--.*?-->\s*)*<svg(?:\s|>)",
    re.IGNORECASE | re.DOTALL,
)


class UnusableAsset(Exception):
    """One figure could not be fetched, and the paper is fine without it.

    Separated from the errors that invalidate a whole run (the total download
    budget, a path escaping the output directory) because those two used to be
    the same thing: any single bad figure aborted the conversion. arXiv answers
    a request for a missing image with the article page — HTML, HTTP 200 — so
    one dead `<img>` in a paper cost the reader the entire document.
    """

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{reason}: {source}")
        self.source = source
        self.reason = reason


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
    # arXiv serves LaTeXML's external vector plots as standalone SVG files.
    # Check the bounded document prefix rather than trusting Content-Type;
    # XML declarations and comments may precede the root element.
    if _SVG_ROOT.match(data[:8192]):
        return "image/svg+xml"
    return None


# A figure with few distinct colours is a plot, a diagram or a table shot:
# sharp text and thin strokes, which lossy coding smears and which lossless
# WebP happens to compress enormously. Everything above the threshold is
# photographic — screenshots, qualitative examples — where the opposite holds.
LINE_ART_MAX_COLORS = 1000
PHOTO_QUALITY = 90
# libwebp refuses anything larger; such a figure keeps its original encoding.
WEBP_MAX_DIMENSION = 16383


def recompress_to_webp(data: bytes) -> tuple[bytes, str] | None:
    """Re-encode a raster image as WebP, or None to keep the original bytes.

    Papers ship figures as arXiv's LaTeX pipeline produced them, which is
    routinely 16-bit-per-channel RGBA PNG — a depth no display, PDF or paper
    workflow can show, at twice the bytes. Re-encoding is worth a lot: line art
    goes lossless and still collapses, photographs take q90 and lose nothing a
    reader can see.
    """
    import io

    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            # An animated source has frames WebP would silently flatten away.
            if getattr(image, "n_frames", 1) > 1:
                return None
            if max(image.size) > WEBP_MAX_DIMENSION:
                return None
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            # getcolors returns None once the image passes the cap, which is
            # exactly the line-art/photograph split we want.
            lossless = image.getcolors(maxcolors=LINE_ART_MAX_COLORS) is not None
            buffer = io.BytesIO()
            if lossless:
                image.save(buffer, format="WEBP", lossless=True, method=6)
            else:
                image.save(buffer, format="WEBP", quality=PHOTO_QUALITY, method=6)
    except Exception:
        # A figure we cannot decode is still a figure; ship it untouched.
        return None
    encoded = buffer.getvalue()
    # Re-encoding is an optimization, never an obligation.
    return (encoded, "image/webp") if encoded and len(encoded) < len(data) else None


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
    """Resolve an image URL against the document's base URL.

    LaTeXML mistakes dotted source directories such as
    ``Figures.HandDraw/chameleon.png`` for hostnames and arXiv publishes the
    result as ``https://Figures.HandDraw/chameleon.png``. The files still live
    beside the paper HTML. Restore that specific malformed shape before the
    normal URL join so the asset remains on arXiv's trusted origin.
    """
    parsed = urlparse(src)
    host_parts = parsed.netloc.split(".")
    if (
        parsed.scheme == "https"
        and len(host_parts) >= 2
        and host_parts[0] == "Figures"
        and any(character.isupper() for character in host_parts[-1])
        and urlparse(source_url).hostname in TRUSTED_HOSTS
    ):
        relative = parsed.netloc + parsed.path
        if parsed.query:
            relative += f"?{parsed.query}"
        if parsed.fragment:
            relative += f"#{parsed.fragment}"
        return urljoin(f"{source_url.rstrip('/')}/", relative)
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
    # Figures download concurrently up to this many at once. Serial downloads
    # made the asset stage scale with figure count times arXiv's latency — a
    # 44-figure paper spent most of its conversion just waiting on round
    # trips. Modest on purpose: this parallelism hits arXiv itself.
    max_concurrency: int = 8


class AssetMaterializer:
    """Materialize paper images and map their source references to local paths."""

    def __init__(self, output_file: Path, *, limits: AssetLimits = AssetLimits(), transport: httpx.AsyncBaseTransport | None = None, compress: bool = False) -> None:
        self.output_file = output_file.resolve()
        self.assets_dir = self.output_file.parent / f"{self.output_file.stem}_assets"
        self.limits = limits
        self.transport = transport
        if compress:
            # Fail here rather than quietly shipping uncompressed figures: the
            # caller asked for this and has no other way to find out.
            try:
                import PIL.Image  # noqa: F401
            except ImportError as exc:  # pragma: no cover - packaging guard
                raise ValueError("Compressing assets requires Pillow") from exc
        self.compress = compress
        self._references: dict[str, str] = {}
        self._inline_entries: list[dict[str, object]] = []
        self._inline_size = 0

    def __call__(self, source_url: str) -> str:
        return self._references.get(source_url, source_url)

    def materialize_inline_svg(self, svg: str, source_id: str | None = None) -> str:
        """Store one LaTeXML inline picture as a standalone SVG asset.

        arXiv emits TikZ/PGF figures as inline ``svg.ltx_picture`` elements,
        not remote ``img`` URLs. They still belong in the same verified paper
        bundle, but vector bytes should bypass raster sniffing and WebP
        compression so plots, labels and thin strokes remain exact.
        """
        data = svg.encode("utf-8")
        if len(data) > self.limits.max_file_bytes:
            raise ValueError("Inline SVG exceeds the per-file limit")
        if len(self._inline_entries) >= self.limits.max_count:
            raise ValueError(f"Asset count exceeds limit ({self.limits.max_count})")
        if self._inline_size + len(data) > self.limits.max_total_bytes:
            raise ValueError("Assets exceed total download limit")

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        root = self.assets_dir.resolve()
        digest = hashlib.sha256(data).hexdigest()
        index = len(self._inline_entries) + 1
        name = f"figure-inline-{index:03d}-{digest[:12]}.svg"
        destination = (root / name).resolve()
        if root not in destination.parents:
            raise ValueError("Asset path escapes output directory")
        destination.write_bytes(data)
        relative = destination.relative_to(self.output_file.parent).as_posix()
        source = f"inline-svg:{source_id or digest[:12]}"
        self._inline_entries.append(
            {
                "source": source,
                "resolved_source": source,
                "path": relative,
                "type": "image/svg+xml",
                "size": len(data),
                "sha256": digest,
            }
        )
        self._inline_size += len(data)
        return relative

    async def materialize(self, source_urls: list[str]) -> None:
        unique = list(dict.fromkeys(source_urls))
        if len(unique) + len(self._inline_entries) > self.limits.max_count:
            raise ValueError(f"Asset count exceeds limit ({self.limits.max_count})")
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        root = self.assets_dir.resolve()
        total = self._inline_size
        semaphore = asyncio.Semaphore(self.limits.max_concurrency)

        async def fetch(source: str) -> tuple[bytes, str, str]:
            nonlocal total
            # The semaphore covers only the network wait; validation and
            # re-encoding release the slot so downloads keep flowing while a
            # figure is being compressed on a worker thread.
            async with semaphore:
                try:
                    response, final_url = await self._get_following_safe_redirects(client, source)
                except UnusableAsset:
                    raise
                except httpx.HTTPError as error:
                    raise UnusableAsset(source, f"Could not fetch asset ({error})") from error
                except ValueError as error:
                    # An untrusted host or a redirect that went nowhere: about
                    # this one URL, not about the paper.
                    raise UnusableAsset(source, str(error)) from error
            data = response.content
            if len(data) > self.limits.max_file_bytes:
                raise UnusableAsset(source, "Asset exceeds the per-file limit")
            # The running total is only advisory under concurrency: requests
            # already in flight may finish after the budget is crossed, so the
            # overshoot is bounded by max_concurrency times the file limit.
            total += len(data)
            if total > self.limits.max_total_bytes:
                raise ValueError("Assets exceed total download limit")
            content_type = sniff_image_type(data)
            if content_type is None:
                declared = response.headers.get("content-type", "").split(";", 1)[0].lower()
                raise UnusableAsset(
                    source,
                    f"Not a supported image (declared {declared or 'nothing'})",
                )
            # After the signature check, so only bytes already proven to be
            # an image are ever handed to the decoder. The manifest then
            # describes what is on disk, not what arrived. On a thread because
            # WebP method=6 is deliberately slow — encoded inline it would
            # stall the event loop and serialize the downloads again.
            if self.compress and content_type != "image/svg+xml":
                recompressed = await asyncio.to_thread(recompress_to_webp, data)
                if recompressed is not None:
                    data, content_type = recompressed
            return data, content_type, final_url

        timeout = httpx.Timeout(self.limits.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=self.transport) as client:
            results = await asyncio.gather(*(fetch(source) for source in unique), return_exceptions=True)
        # Surface the first failure in source order, not completion order, so
        # a broken paper reports the same error on every run. Only the failures
        # that invalidate the whole run stop it; a figure that could not be
        # fetched is recorded and left out, and its Markdown keeps pointing at
        # the original URL.
        skipped: list[dict[str, object]] = []
        for source, result in zip(unique, results):
            if isinstance(result, UnusableAsset):
                logger.warning("Skipping figure %s: %s", result.source, result.reason)
                skipped.append({"source": source, "reason": result.reason})
            elif isinstance(result, BaseException):
                raise result
        # Everything ordering-sensitive — figure numbering, reference mapping,
        # the manifest — happens down here in source order, exactly as the
        # serial loop did it.
        entries = list(self._inline_entries)
        for index, (source, result) in enumerate(zip(unique, results), 1):
            if isinstance(result, UnusableAsset):
                continue
            data, content_type, final_url = result
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
        manifest = {"schema_version": MANIFEST_SCHEMA, "converter": CONVERTER, "converter_version": "0.1.0", "assets": entries, "skipped": skipped}
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
