"""Convert arXiv HTML to Markdown with a custom serializer."""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
    from bs4.element import NavigableString, Tag
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("BeautifulSoup4 is required for HTML parsing (pip install beautifulsoup4).") from exc


_EQUATION_TABLE_RE = re.compile(r"ltx_equationgroup|ltx_eqn_align|ltx_eqn_table")
_MAX_TABLE_COLSPAN = 1_000
_MAX_TABLE_ROWSPAN = 65_534


def convert_html_to_markdown(html: str, *, remove_refs: bool = False, remove_toc: bool = False) -> str:
    """Convert arXiv HTML into Markdown."""
    soup = BeautifulSoup(html, "html.parser")
    toc_markdown = None
    toc_nav = soup.find("nav", class_=re.compile(r"ltx_TOC"))
    if toc_nav and not remove_toc:
        toc_markdown = _serialize_toc(toc_nav)

    _strip_unwanted_elements(soup)
    if remove_refs:
        for ref in soup.find_all("section", class_=re.compile(r"ltx_bibliography")):
            ref.decompose()

    convert_all_mathml_to_latex(soup)
    fix_tabular_tables(soup)

    root = _find_document_root(soup)
    title_tag = root.find("h1", class_=re.compile(r"ltx_title_document"))
    authors_tag = root.find("div", class_=re.compile(r"ltx_authors"))
    abstract_tag = root.find("div", class_=re.compile(r"ltx_abstract"))

    blocks: list[str] = []
    if title_tag:
        blocks.append(f"# {_normalize_text(title_tag.get_text(' ', strip=True))}")
    if authors_tag:
        authors_text = _normalize_text(authors_tag.get_text(" ", strip=True))
        if authors_text:
            blocks.append(f"Authors: {authors_text}")
    if toc_markdown:
        blocks.append("## Contents\n" + toc_markdown)
    if abstract_tag:
        blocks.extend(_serialize_abstract(abstract_tag))

    for tag in (title_tag, authors_tag, abstract_tag):
        if tag:
            tag.decompose()

    blocks.extend(_serialize_children(root))

    return "\n\n".join(block for block in blocks if block).strip()


def convert_fragment_to_markdown(html: str, *, remove_inline_citations: bool = False, base_url: str | None = None, asset_materializer: Callable[[str], str] | None = None) -> str:
    """Convert an HTML fragment into Markdown without title/author/abstract handling.

    Parameters
    ----------
    html : str
        The HTML fragment to convert.
    remove_inline_citations : bool
        If True, completely remove inline citation links. If False (default),
        citation links are converted to plain text (URL stripped).
    base_url : str | None
        Base URL to resolve relative image paths against. When provided,
        relative ``<img src>`` attributes are converted to absolute URLs.
    """
    soup = BeautifulSoup(html, "html.parser")
    _strip_unwanted_elements(soup)
    _convert_equation_tables(soup)
    convert_all_mathml_to_latex(soup)
    fix_tabular_tables(soup)
    if base_url:
        _resolve_image_urls(soup, base_url)
    if asset_materializer:
        for image in soup.find_all("img", src=True):
            image["src"] = asset_materializer(str(image["src"]))
    blocks = _serialize_children(soup, remove_inline_citations=remove_inline_citations)
    return "\n\n".join(block for block in blocks if block).strip()


def _find_document_root(soup: BeautifulSoup) -> Tag:
    root = soup.find("article", class_=re.compile(r"ltx_document"))
    if root:
        return root
    if soup.body:
        return soup.body
    return soup


def _strip_unwanted_elements(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(["script", "style", "noscript", "link", "meta"]):
        tag.decompose()
    for tag in soup.select("nav.ltx_page_navbar, nav.ltx_TOC"):
        tag.decompose()
    for tag in soup.select("button.sr-only, div.package-alerts, div.ltx_pagination, footer"):
        tag.decompose()


def convert_all_mathml_to_latex(root: BeautifulSoup) -> None:
    for math in root.find_all("math"):
        annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
        if annotation and annotation.text:
            latex_source = annotation.text.strip()
            latex_source = re.sub(r"(?<!\\)%", "", latex_source)
            latex_source = re.sub(r"\\([_^])", r"\1", latex_source)
            latex_source = re.sub(r"\\(?=[\[\]])", "", latex_source)
            display = math.get("display") == "block"
            math.replace_with(f"$$\n{latex_source}\n$$" if display else f"${latex_source}$")
        elif math.get("alttext"):
            math.replace_with(f"${math['alttext'].strip()}$")
        else:
            math.replace_with(math.get_text(" ", strip=True))


def fix_tabular_tables(root: BeautifulSoup) -> None:
    tables = root.find_all("table", class_=re.compile(r"ltx_tabular"))
    for table in tables:
        _repair_rotated_row_group_spans(table)
        _remove_all_attributes(table)
        for child in table.find_all(["tbody", "thead", "tfoot", "tr", "td", "th"]):
            spans = {}
            if child.name in {"td", "th"}:
                for attribute in ("rowspan", "colspan"):
                    span = _table_span(child, attribute)
                    if span != 1:
                        spans[attribute] = str(span)
            child.attrs = spans


def _repair_rotated_row_group_spans(table: Tag) -> None:
    """Extend an under-counted vertical row label over trailing data rows.

    LaTeX tables commonly rotate a ``multirow`` label in their first column.
    Some papers under-count that multirow while still leaving an empty first
    cell on the remaining data rows. Browsers make the visual grouping fairly
    clear, but the literal HTML rowspan ends early. Treat those empty cells as
    continuation placeholders unless a top border starts a new row group.
    """
    for cell in table.find_all(["td", "th"], rowspan=True):
        rowspan = _table_span(cell, "rowspan")
        if rowspan <= 1 or not cell.find(
            style=re.compile(r"rotate\(\s*-?90deg\s*\)", re.IGNORECASE)
        ):
            continue
        row = cell.find_parent("tr")
        row_group = row.parent if row else None
        if not row or not isinstance(row_group, Tag):
            continue
        rows = row_group.find_all("tr", recursive=False)
        try:
            row_index = rows.index(row)
        except ValueError:
            continue
        row_cells = row.find_all(["td", "th"], recursive=False)
        if not row_cells or row_cells[0] is not cell:
            continue

        placeholders: list[Tag] = []
        for continuation in rows[row_index + rowspan :]:
            continuation_cells = continuation.find_all(["td", "th"], recursive=False)
            if not continuation_cells:
                break
            placeholder = continuation_cells[0]
            classes = set(placeholder.get("class", []))
            if "ltx_border_t" in classes or placeholder.get_text(" ", strip=True):
                break
            if not any(candidate.get_text(" ", strip=True) for candidate in continuation_cells[1:]):
                break
            placeholders.append(placeholder)

        if not placeholders:
            continue
        cell["rowspan"] = str(rowspan + len(placeholders))
        for placeholder in placeholders:
            placeholder.decompose()


def _resolve_image_urls(root: BeautifulSoup, base_url: str) -> None:
    """Resolve relative ``<img src>`` attributes to absolute URLs."""
    for img in root.find_all("img"):
        src = img.get("src")
        if src and not src.startswith(("http://", "https://", "data:")):
            img["src"] = urljoin(base_url, src)


def _remove_all_attributes(tag: Tag) -> None:
    tag.attrs = {}


def _table_span(cell: Tag, attribute: str) -> int:
    try:
        span = int(str(cell.get(attribute, 1)))
    except (TypeError, ValueError):
        return 1
    if attribute == "rowspan" and span == 0:
        return 0
    maximum = _MAX_TABLE_ROWSPAN if attribute == "rowspan" else _MAX_TABLE_COLSPAN
    return min(max(1, span), maximum)


def _escape_table_pipes(text: str) -> str:
    return re.sub(
        r"(\\*)\|",
        lambda match: match.group(1) + (r"\|" if len(match.group(1)) % 2 == 0 else "|"),
        text,
    )


def _serialize_children(container: Tag, *, remove_inline_citations: bool = False) -> list[str]:
    blocks: list[str] = []
    for child in container.children:
        if isinstance(child, NavigableString):
            continue
        if not isinstance(child, Tag):
            continue
        blocks.extend(_serialize_block(child, remove_inline_citations=remove_inline_citations))
    return blocks


def _serialize_block(tag: Tag, *, remove_inline_citations: bool = False) -> list[str]:
    if "arxiv2md_equations" in tag.get("class", []):
        return [tag.get_text()]

    if tag.name in {"section", "article", "div", "span"}:
        return _serialize_children(tag, remove_inline_citations=remove_inline_citations)

    if tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(tag.name[1])
        heading = _normalize_text(tag.get_text(" ", strip=True))
        if not heading:
            return []
        return [f"{'#' * level} {heading}"]

    if tag.name == "p":
        paragraph = _serialize_paragraph(tag, remove_inline_citations=remove_inline_citations)
        return [paragraph] if paragraph else []

    if tag.name in {"ul", "ol"}:
        lines = _serialize_list(tag, remove_inline_citations=remove_inline_citations)
        return ["\n".join(lines)] if lines else []

    if tag.name == "figure":
        figure = _serialize_figure(tag, remove_inline_citations=remove_inline_citations)
        return [figure] if figure else []

    if tag.name == "table":
        table_md = _serialize_table(tag, remove_inline_citations=remove_inline_citations)
        return [table_md] if table_md else []

    if tag.name == "blockquote":
        content = _normalize_text(_serialize_inline(tag, remove_inline_citations=remove_inline_citations))
        if not content:
            return []
        return ["> " + content]

    if tag.name == "br":
        return []

    return _serialize_children(tag, remove_inline_citations=remove_inline_citations)


def _serialize_abstract(tag: Tag) -> list[str]:
    blocks = ["## Abstract"]
    paragraphs = tag.find_all("p")
    if not paragraphs:
        content = _normalize_text(tag.get_text(" ", strip=True))
        if content:
            blocks.append(content)
        return blocks

    for paragraph in paragraphs:
        text = _serialize_paragraph(paragraph)
        if text:
            blocks.append(text)
    return blocks


def _serialize_paragraph(tag: Tag, *, remove_inline_citations: bool = False) -> str:
    content = _serialize_inline(tag, remove_inline_citations=remove_inline_citations)
    content = _cleanup_inline_text(content)
    return content


def _is_citation_link(href: str | None) -> bool:
    """Check if a link is a citation reference (e.g., #bib.bib7)."""
    if not href:
        return False
    return "#bib." in href or href.startswith("#bib")


def _is_internal_paper_link(href: str | None) -> bool:
    """Check if a link is an internal paper section reference (e.g., arxiv.org/html/...#S2.SS1)."""
    if not href:
        return False
    return "arxiv.org/html/" in href and "#" in href and "#bib" not in href


def _serialize_inline(node: Tag | NavigableString, *, remove_inline_citations: bool = False) -> str:
    if isinstance(node, NavigableString):
        return str(node)

    if node.name == "br":
        return "\n"

    if node.name in {"em", "i"}:
        return f"*{_serialize_children_inline(node, remove_inline_citations=remove_inline_citations)}*"

    if node.name in {"strong", "b"}:
        return f"**{_serialize_children_inline(node, remove_inline_citations=remove_inline_citations)}**"

    if node.name == "a":
        text = _serialize_children_inline(node, remove_inline_citations=remove_inline_citations).strip()
        href = node.get("href")
        # Handle citation links specially
        if _is_citation_link(href):
            if remove_inline_citations:
                return ""  # Completely remove citation
            return text  # Keep text only, strip URL
        # Handle internal paper links (section references)
        if remove_inline_citations and _is_internal_paper_link(href):
            return text  # Keep text only, strip URL
        # Regular links: keep full markdown link
        if href:
            return f"[{text or href}]({href})"
        return text

    if node.name == "sup":
        text = _serialize_children_inline(node, remove_inline_citations=remove_inline_citations).strip()
        return f"^{text}" if text else ""

    if node.name == "cite":
        if remove_inline_citations and "ltx_cite" in node.get("class", []):
            return ""
        return _serialize_children_inline(node, remove_inline_citations=remove_inline_citations)

    if node.name == "math":
        text = node.get_text(" ", strip=True)
        return f"${text}$" if text else ""

    if "ltx_note" in node.get("class", []):
        text = _normalize_text(_serialize_children_inline(node, remove_inline_citations=remove_inline_citations))
        return f"({text})" if text else ""

    return _serialize_children_inline(node, remove_inline_citations=remove_inline_citations)


def _serialize_children_inline(tag: Tag, *, remove_inline_citations: bool = False) -> str:
    return "".join(_serialize_inline(child, remove_inline_citations=remove_inline_citations) for child in tag.children)


def _cleanup_inline_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def _serialize_list(list_tag: Tag, indent: int = 0, *, remove_inline_citations: bool = False) -> list[str]:
    lines: list[str] = []
    for item in list_tag.find_all("li", recursive=False):
        item_text_parts: list[str] = []
        nested_lists: list[Tag] = []
        for child in item.children:
            if isinstance(child, Tag) and child.name in {"ul", "ol"}:
                nested_lists.append(child)
            else:
                item_text_parts.append(_serialize_inline(child, remove_inline_citations=remove_inline_citations))
        item_text = _cleanup_inline_text("".join(item_text_parts))
        prefix = "  " * indent + "- "
        lines.append(prefix + item_text if item_text else prefix.rstrip())
        for nested in nested_lists:
            lines.extend(_serialize_list(nested, indent + 1, remove_inline_citations=remove_inline_citations))
    return lines


def _serialize_toc(toc_nav: Tag) -> str:
    list_tag = toc_nav.find("ol")
    if not list_tag:
        return ""
    lines = _serialize_list(list_tag)
    return "\n".join(lines)


def _serialize_table(table: Tag, *, remove_inline_citations: bool = False) -> str:
    classes = " ".join(table.get("class", []))
    if _EQUATION_TABLE_RE.search(classes):
        eqn_text = _normalize_text(table.get_text(" ", strip=True))
        if not eqn_text:
            return ""
        return f"$$ {eqn_text} $$"

    row_groups: list[list[Tag]] = []
    loose_rows: list[Tag] = []
    for child in table.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "tr":
            loose_rows.append(child)
            continue
        if child.name not in {"tbody", "thead", "tfoot"}:
            continue
        if loose_rows:
            row_groups.append(loose_rows)
            loose_rows = []
        section_rows = child.find_all("tr", recursive=False)
        if section_rows:
            row_groups.append(section_rows)
    if loose_rows:
        row_groups.append(loose_rows)

    rows: list[list[str]] = []
    # GFM has no merged-cell syntax. Expand each merged cell over every logical
    # slot it covers and repeat its text there. Repetition makes row and column
    # meaning explicit to both Markdown renderers and text-only consumers.
    active_rowspans: dict[int, tuple[str, int]] = {}
    for row_group in row_groups:
        active_rowspans = {}
        for row_index, row in enumerate(row_group):
            cells = row.find_all(["th", "td"], recursive=False)
            values = {
                column: text
                for column, (text, _remaining_rows) in active_rowspans.items()
            }
            column = 0
            new_rowspans: dict[int, tuple[str, int]] = {}
            for cell in cells:
                colspan = _table_span(cell, "colspan")
                while column in values:
                    column += 1
                cell_text = _cleanup_inline_text(
                    _serialize_inline(
                        cell,
                        remove_inline_citations=remove_inline_citations,
                    )
                ).replace("\n", "<br>")
                cell_text = _escape_table_pipes(cell_text)
                owned_columns = [
                    occupied_column
                    for occupied_column in range(column, column + colspan)
                    if occupied_column not in values
                ]
                for occupied_column in owned_columns:
                    values[occupied_column] = cell_text

                rowspan = _table_span(cell, "rowspan")
                remaining_rows = (
                    len(row_group) - row_index - 1
                    if rowspan == 0
                    else rowspan - 1
                )
                if remaining_rows:
                    for occupied_column in owned_columns:
                        new_rowspans[occupied_column] = (cell_text, remaining_rows)
                column += colspan

            if values:
                width = max(values) + 1
                rows.append([values.get(column, "") for column in range(width)])
            active_rowspans = {
                occupied_column: (text, remaining_rows - 1)
                for occupied_column, (text, remaining_rows) in active_rowspans.items()
                if remaining_rows > 1
            }
            active_rowspans.update(new_rowspans)

    if not rows:
        return ""

    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    header = normalized[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _convert_equation_tables(root: BeautifulSoup) -> None:
    """Serialize LaTeXML equation groups before generic MathML replacement."""
    for table in root.find_all("table", class_=_EQUATION_TABLE_RE):
        rendered: list[str] = []
        converted_rows = 0
        if table.get("id"):
            rendered.append(f'<a id="{table["id"]}"></a>')
        for row in table.find_all("tr"):
            number = row.find(class_=re.compile(r"ltx_tag|ltx_eqn_number"))
            cells: list[str] = []
            for cell in row.find_all(["td", "th"], recursive=False):
                if number and (cell is number or number in cell.descendants):
                    continue
                parts = [_extract_math_tex(math) for math in cell.find_all("math")]
                tex = " ".join(part for part in parts if part)
                if tex:
                    cells.append(tex)
            if not cells:
                continue
            converted_rows += 1
            tex = cells[0] if len(cells) == 1 else "\\begin{aligned}" + " & ".join(cells) + "\\end{aligned}"
            row_group = row.find_parent("tbody")
            row_id = row.get("id") or (row_group.get("id") if row_group else None)
            suffix = number.get_text(" ", strip=True) if number else ""
            equation_number = suffix.removeprefix("(").removesuffix(")").strip()
            if equation_number:
                tex += f"\\tag{{{equation_number}}}"
            anchor = f'<a id="{row_id}"></a>\n' if row_id else ""
            rendered.append(f"{anchor}$$\n{tex}\n$$")
        if not converted_rows:
            continue
        holder = root.new_tag("div")
        holder["class"] = ["arxiv2md_equations"]
        holder.string = "\n\n".join(rendered)
        table.replace_with(holder)


def _extract_math_tex(math: Tag) -> str:
    annotation = math.find("annotation", attrs={"encoding": "application/x-tex"})
    value = annotation.get_text(strip=True) if annotation else str(math.get("alttext", "")).strip()
    return _strip_math_delimiters(value)


def _strip_math_delimiters(tex: str) -> str:
    """Remove outer TeX delimiters so display output has exactly one pair."""
    value = tex.strip()
    if value.startswith("\\[") and value.endswith("\\]"):
        return value[2:-2].strip()
    if value.startswith("$$") and value.endswith("$$"):
        return value[2:-2].strip()
    if value.startswith("$") and value.endswith("$"):
        return value[1:-1].strip()
    return value


def _serialize_figure(figure: Tag, *, remove_inline_citations: bool = False) -> str:
    # Check if this is a table figure (ltx_table class)
    figure_classes = " ".join(figure.get("class", []))
    is_table_figure = "ltx_table" in figure_classes

    flex_figure = _serialize_flex_figure(
        figure,
        remove_inline_citations=remove_inline_citations,
    )
    if flex_figure:
        return flex_figure

    caption_tag = figure.find("figcaption")
    caption = _normalize_text(_serialize_inline(caption_tag, remove_inline_citations=remove_inline_citations)) if caption_tag else ""

    lines = []

    if is_table_figure:
        # Handle table figures - find and serialize the embedded table
        # Note: fix_tabular_tables strips attributes, so search for any table element
        table = figure.find("table")
        if table:
            table_md = _serialize_table(table, remove_inline_citations=remove_inline_citations)
            if caption:
                lines.append(f"**{caption}**")
            if table_md:
                lines.append(table_md)
        elif caption:
            # Fallback if no table found but has caption
            lines.append(f"Table: {caption}")
    else:
        # Handle regular image figures
        for img in figure.find_all("img"):
            src = img.get("src")
            if src:
                lines.append(f"![{img.get('alt') or caption or 'Figure'}]({src})")
        if caption:
            lines.append(f"*{caption}*")

    body = "\n\n".join(lines).strip()
    figure_id = figure.get("id")
    return (f'<a id="{figure_id}"></a>\n\n' + body) if figure_id and body else body


def _serialize_flex_figure(figure: Tag, *, remove_inline_citations: bool = False) -> str:
    """Preserve LaTeXML's multi-panel figure rows as structured MDX.

    ``ltx_flex_size_N`` means a cell occupies one Nth of the figure width,
    while ``ltx_flex_break`` forces a new row. Plain Markdown has no figure
    layout syntax, so emit explicit structural components rather than flattening
    every panel into an unrelated image paragraph. Consumers that understand
    the components can recreate the source rows; text-only consumers still see
    ordinary Markdown images and captions inside them.
    """
    flex_containers = [
        child
        for child in figure.children
        if isinstance(child, Tag) and "ltx_flex_figure" in child.get("class", [])
    ]
    if not flex_containers:
        return ""

    body: list[str] = []
    for child in figure.children:
        if not isinstance(child, Tag):
            continue
        if "ltx_flex_figure" in child.get("class", []):
            for cells in _flex_figure_rows(child):
                panels: list[tuple[str, int]] = []
                for cell, denominator in cells:
                    panel = _serialize_flex_panel(
                        cell,
                        remove_inline_citations=remove_inline_citations,
                    )
                    if panel:
                        panels.append((panel, denominator))
                if not panels:
                    continue
                columns = " ".join(str(denominator) for _panel, denominator in panels)
                body.append(
                    f'<PaperFigureRow columns="{columns}">\n\n'
                    + "\n\n".join(panel for panel, _denominator in panels)
                    + "\n\n</PaperFigureRow>"
                )
        elif child.name == "figcaption":
            caption = _figure_caption(child, remove_inline_citations=remove_inline_citations)
            if caption:
                body.append(f"*{caption}*")

    if not body:
        return ""
    figure_id = figure.get("id")
    id_attr = f' id="{html.escape(str(figure_id), quote=True)}"' if figure_id else ""
    return f"<PaperFigure{id_attr}>\n\n" + "\n\n".join(body) + "\n\n</PaperFigure>"


def _flex_figure_rows(container: Tag) -> list[list[tuple[Tag, int]]]:
    """Split flex cells at explicit breaks and at the browser's wrap point."""
    groups: list[list[Tag]] = []
    current: list[Tag] = []
    for child in container.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class", [])
        if "ltx_flex_break" in classes:
            if current:
                groups.append(current)
                current = []
            continue
        if "ltx_flex_cell" in classes:
            current.append(child)
    if current:
        groups.append(current)

    rows: list[list[tuple[Tag, int]]] = []
    for group in groups:
        fallback = max(1, len(group))
        row: list[tuple[Tag, int]] = []
        occupied = 0.0
        for cell in group:
            denominator = _flex_cell_denominator(cell) or fallback
            width = 1 / denominator
            if row and occupied + width > 1.000_001:
                rows.append(row)
                row = []
                occupied = 0.0
            row.append((cell, denominator))
            occupied += width
        if row:
            rows.append(row)
    return rows


def _flex_cell_denominator(cell: Tag) -> int | None:
    for class_name in cell.get("class", []):
        match = re.fullmatch(r"ltx_flex_size_(\d+)", class_name)
        if match:
            value = int(match.group(1))
            if value > 0:
                return value
    return None


def _serialize_flex_panel(cell: Tag, *, remove_inline_citations: bool = False) -> str:
    panel = next(
        (
            child
            for child in cell.children
            if isinstance(child, Tag)
            and child.name == "figure"
            and "ltx_figure_panel" in child.get("class", [])
        ),
        cell,
    )
    content: list[str] = []
    for element in panel.find_all(["img", "figcaption"]):
        if element.name == "img":
            src = element.get("src")
            if not src:
                continue
            alt = str(element.get("alt") or "Figure")
            content.append(f"![{alt}]({src})")
        else:
            caption = _figure_caption(element, remove_inline_citations=remove_inline_citations)
            if caption:
                content.append(f"*{caption}*")
    if not content:
        return ""

    panel_id = panel.get("id")
    id_attr = f' id="{html.escape(str(panel_id), quote=True)}"' if panel_id else ""
    return (
        f"<PaperFigurePanel{id_attr}>\n\n"
        + "\n\n".join(content)
        + "\n\n</PaperFigurePanel>"
    )


def _figure_caption(caption: Tag, *, remove_inline_citations: bool = False) -> str:
    return _normalize_text(
        _serialize_inline(
            caption,
            remove_inline_citations=remove_inline_citations,
        )
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
