"""Tests for Markdown serialization."""

from __future__ import annotations

from arxiv2md.markdown import convert_fragment_to_markdown


def test_math_and_tables_render() -> None:
    html = """
    <div class="ltx_para"><p>Equation <math>
        <annotation encoding="application/x-tex">x+y</annotation>
    </math></p></div>
    <table class="ltx_tabular">
        <tr><th>A</th><th>B</th></tr>
        <tr><td>1</td><td>2</td></tr>
    </table>
    <table class="ltx_equationgroup">
        <tr><td>E = mc^2 (1)</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    assert "$x+y$" in markdown
    assert "| A | B |" in markdown
    assert "| 1 | 2 |" in markdown
    assert "$$" in markdown
    assert "E = mc^2" in markdown


def test_table_with_tbody() -> None:
    """Test that tables with tbody/thead/tfoot structure are correctly converted."""
    html = """
    <table class="ltx_tabular">
        <tbody>
            <tr><th>Model</th><th>Accuracy</th></tr>
            <tr><td>Llama-7B</td><td>70.12</td></tr>
            <tr><td>Llama-13B</td><td>72.39</td></tr>
        </tbody>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    # Should contain table structure
    assert "| Model | Accuracy |" in markdown
    assert "| --- | --- |" in markdown
    assert "| Llama-7B | 70.12 |" in markdown
    assert "| Llama-13B | 72.39 |" in markdown


def test_table_with_thead_tbody() -> None:
    """Test that tables with thead and tbody are correctly converted."""
    html = """
    <table class="ltx_tabular">
        <thead>
            <tr><th>Method</th><th>Result</th></tr>
        </thead>
        <tbody>
            <tr><td>Prune SW</td><td>0.0%</td></tr>
            <tr><td>Prune Non-SW</td><td>68.5%</td></tr>
        </tbody>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    # Should contain table structure
    assert "| Method | Result |" in markdown
    assert "| Prune SW | 0.0% |" in markdown
    assert "| Prune Non-SW | 68.5% |" in markdown


def test_table_inside_figure() -> None:
    """Test that tables wrapped in figure elements (ltx_table) are correctly converted."""
    html = """
    <figure class="ltx_table" id="S3.T1">
        <table class="ltx_tabular ltx_centering ltx_guessed_headers ltx_align_middle">
            <thead class="ltx_thead">
                <tr class="ltx_tr">
                    <th class="ltx_td ltx_align_left ltx_th ltx_th_column">Model</th>
                    <th class="ltx_td ltx_align_center ltx_th ltx_th_column">Arc-c</th>
                    <th class="ltx_td ltx_align_center ltx_th ltx_th_column">Arc-e</th>
                </tr>
            </thead>
            <tbody class="ltx_tbody">
                <tr class="ltx_tr">
                    <td class="ltx_td ltx_align_left">Original</td>
                    <td class="ltx_td ltx_align_center">41.81</td>
                    <td class="ltx_td ltx_align_center">75.29</td>
                </tr>
                <tr class="ltx_tr">
                    <td class="ltx_td ltx_align_left">Prune SW</td>
                    <td class="ltx_td ltx_align_center">19.80</td>
                    <td class="ltx_td ltx_align_center">39.60</td>
                </tr>
            </tbody>
        </table>
        <figcaption class="ltx_caption ltx_centering">
            <span class="ltx_tag ltx_tag_table">Table 1: </span>
            <span class="ltx_text ltx_font_bold">Super Weight Importance</span>.
            Pruning the super weight significantly impairs quality.
        </figcaption>
    </figure>
    """

    markdown = convert_fragment_to_markdown(html)

    # Should contain the caption
    assert "Table 1:" in markdown
    assert "Super Weight Importance" in markdown
    # Should contain the actual table data
    assert "| Model | Arc-c | Arc-e |" in markdown
    assert "| Original | 41.81 | 75.29 |" in markdown
    assert "| Prune SW | 19.80 | 39.60 |" in markdown


def test_table_repeats_rowspan_and_colspan_values_over_the_logical_grid() -> None:
    """Merged HTML cells keep their meaning in every covered Markdown slot."""
    html = """
    <table class="ltx_tabular ltx_align_middle">
        <tr>
            <th rowspan="2" class="ltx_td">Category</th>
            <th rowspan="2" class="ltx_td">Model</th>
            <th colspan="2" class="ltx_td">Score</th>
        </tr>
        <tr><th>Easy</th><th>Hard</th></tr>
        <tr><td rowspan="2">Base</td><td>A</td><td>1</td><td>2</td></tr>
        <tr><td>B</td><td>3</td><td>4</td></tr>
        <tr><td></td><td>C</td><td>5</td><td>6</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    assert markdown == "\n".join(
        [
            "| Category | Model | Score | Score |",
            "| --- | --- | --- | --- |",
            "| Category | Model | Easy | Hard |",
            "| Base | A | 1 | 2 |",
            "| Base | B | 3 | 4 |",
            "|  | C | 5 | 6 |",
        ]
    )


def test_wide_table_keeps_every_metric_and_variant_in_its_source_column() -> None:
    """Regression for C-RADIOv4 Table 5 (arXiv 2601.17237v1)."""
    html = """
    <table class="ltx_tabular">
        <tr>
            <td></td>
            <th rowspan="2">Model</th>
            <th colspan="8">SA-Co/Gold Instance Segmentation (cgF1)</th>
        </tr>
        <tr>
            <td></td><th>metaclip_nps</th><th>sa1b_nps</th><th>crowded</th>
            <th>fg_food</th><th>fg_sports_equipment</th><th>attributes</th>
            <th>wiki_common</th><th>Avg</th>
        </tr>
        <tr><td></td><td>SAM3</td><td>47.3</td><td>53.7</td><td>61.1</td><td>53.4</td><td>65.5</td><td>54.9</td><td>42.5</td><td>54.1</td></tr>
        <tr><td rowspan="4"><span style="transform: rotate(-90deg)">C-RADIOv4</span></td><td>SO400M-VDT8</td><td>43.0</td><td>44.5</td><td>54.9</td><td>38.4</td><td>38.4</td><td>40.3</td><td>22.2</td><td>40.3</td></tr>
        <tr><td>SO400M-G</td><td>43.8</td><td>45.7</td><td>55.9</td><td>40.1</td><td>39.8</td><td>41.6</td><td>23.1</td><td>41.4</td></tr>
        <tr><td>H-VDT8</td><td>45.2</td><td>48.1</td><td>56.6</td><td>40.3</td><td>45.3</td><td>44.0</td><td>26.2</td><td>43.7</td></tr>
        <tr><td>H-VDT12</td><td>45.6</td><td>48.4</td><td>57.3</td><td>40.2</td><td>46.1</td><td>45.2</td><td>26.7</td><td>44.2</td></tr>
        <tr><td></td><td>H-G</td><td>45.9</td><td>48.8</td><td>57.4</td><td>40.9</td><td>46.5</td><td>45.9</td><td>27.3</td><td>44.7</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)
    lines = markdown.splitlines()

    assert all(line.count("|") == 11 for line in lines)
    assert lines[2] == "|  | Model | metaclip_nps | sa1b_nps | crowded | fg_food | fg_sports_equipment | attributes | wiki_common | Avg |"
    assert lines[4] == "| C-RADIOv4 | SO400M-VDT8 | 43.0 | 44.5 | 54.9 | 38.4 | 38.4 | 40.3 | 22.2 | 40.3 |"
    assert lines[5] == "| C-RADIOv4 | SO400M-G | 43.8 | 45.7 | 55.9 | 40.1 | 39.8 | 41.6 | 23.1 | 41.4 |"
    assert lines[7] == "| C-RADIOv4 | H-VDT12 | 45.6 | 48.4 | 57.3 | 40.2 | 46.1 | 45.2 | 26.7 | 44.2 |"
    assert lines[8] == "| C-RADIOv4 | H-G | 45.9 | 48.8 | 57.4 | 40.9 | 46.5 | 45.9 | 27.3 | 44.7 |"


def test_table_handles_multiple_active_spans_and_real_empty_cells() -> None:
    """Independent spans never shift cells into another logical column."""
    html = """
    <table class="ltx_tabular">
        <tr><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th></tr>
        <tr>
            <td rowspan="3">left</td>
            <td>one</td>
            <td rowspan="2" colspan="2">middle</td>
            <td>right-1</td>
        </tr>
        <tr><td>two</td><td>right-2</td></tr>
        <tr><td colspan="2"></td><td>four</td><td>right-3</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    assert markdown.splitlines() == [
        "| A | B | C | D | E |",
        "| --- | --- | --- | --- | --- |",
        "| left | one | middle | middle | right-1 |",
        "| left | two | middle | middle | right-2 |",
        "| left |  |  | four | right-3 |",
    ]


def test_malformed_overlapping_spans_do_not_shift_following_cells() -> None:
    html = """
    <table class="ltx_tabular">
        <tr><th>A</th><th>B</th><th>C</th></tr>
        <tr><td>left</td><td rowspan="2">middle</td><td>right</td></tr>
        <tr><td colspan="2">wide</td><td>after</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    assert markdown.splitlines() == [
        "| A | B | C |",
        "| --- | --- | --- |",
        "| left | middle | right |",
        "| wide | middle | after |",
    ]


def test_table_rowspan_zero_extends_to_the_end_of_its_row_group() -> None:
    html = """
    <table class="ltx_tabular">
        <thead>
            <tr><th>Group A</th><th>Group B</th><th>Value</th></tr>
        </thead>
        <tbody>
            <tr><td rowspan="0" colspan="2">all</td><td>1</td></tr>
            <tr><td>2</td></tr>
            <tr><td>3</td></tr>
        </tbody>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    assert markdown.splitlines() == [
        "| Group A | Group B | Value |",
        "| --- | --- | --- |",
        "| all | all | 1 |",
        "| all | all | 2 |",
        "| all | all | 3 |",
    ]


def test_table_escapes_pipes_according_to_preceding_backslash_parity() -> None:
    html = r"""
    <table class="ltx_tabular">
        <tr><th colspan="2">zero | one \| two \\| three \\\|</th></tr>
        <tr><td>1</td><td>2</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)

    assert markdown.splitlines() == [
        r"| zero \| one \| two \\\| three \\\| | zero \| one \| two \\\| three \\\| |",
        "| --- | --- |",
        "| 1 | 2 |",
    ]


def test_table_clamps_oversized_spans_and_defaults_invalid_spans_to_one() -> None:
    html = """
    <table class="ltx_tabular">
        <tr><th colspan="1000000000">wide</th></tr>
        <tr><td colspan="invalid">one</td><td colspan="-2">two</td></tr>
    </table>
    """

    markdown = convert_fragment_to_markdown(html)
    lines = markdown.splitlines()

    assert lines[0].count("|") == 1_001
    assert lines[2].startswith("| one | two |")
    assert len(lines[2].split("|")) == 1_002


def test_remove_inline_citations_citep() -> None:
    """Test that parenthetical citations (citep) are fully removed."""
    html = (
        '<p>We study deceptive alignment '
        '<cite class="ltx_cite ltx_citemacro_citep">'
        '(Anthropic, <a class="ltx_ref" href="#bib.bib4">2024</a>; '
        'OpenAI, <a class="ltx_ref" href="#bib.bib29">2024</a>)'
        '</cite> in large models.</p>'
    )

    result = convert_fragment_to_markdown(html, remove_inline_citations=True)
    assert "Anthropic" not in result
    assert "OpenAI" not in result
    assert "2024" not in result
    assert "deceptive alignment" in result
    assert "large models" in result


def test_remove_inline_citations_citet() -> None:
    """Test that textual citations (citet) are fully removed."""
    html = (
        '<p>As shown by '
        '<cite class="ltx_cite ltx_citemacro_citet">'
        'Treutlein et al. (<a class="ltx_ref" href="#bib.bib40">2024</a>)'
        '</cite>, this is important.</p>'
    )

    result = convert_fragment_to_markdown(html, remove_inline_citations=True)
    assert "Treutlein" not in result
    assert "this is important" in result


def test_remove_inline_citations_preserves_when_disabled() -> None:
    """Test that citations are preserved as plain text when removal is disabled."""
    html = (
        '<p>We study '
        '<cite class="ltx_cite ltx_citemacro_citep">'
        '(Anthropic, <a class="ltx_ref" href="#bib.bib4">2024</a>)'
        '</cite> things.</p>'
    )

    result = convert_fragment_to_markdown(html, remove_inline_citations=False)
    assert "Anthropic" in result
    assert "2024" in result


def test_remove_inline_citations_ignores_non_ltx_cite() -> None:
    """Test that plain cite tags without ltx_cite class are not removed."""
    html = '<p>See <cite>A Book Title</cite> for details.</p>'

    result = convert_fragment_to_markdown(html, remove_inline_citations=True)
    assert "A Book Title" in result


def test_resolve_relative_image_urls() -> None:
    """Test that relative image paths are resolved to absolute URLs."""
    html = """
    <figure>
        <img src="extracted/figures/fig1.png" alt="Architecture diagram"/>
        <figcaption>Figure 1: System overview</figcaption>
    </figure>
    """

    result = convert_fragment_to_markdown(html, base_url="https://arxiv.org/html/2501.11120v1")
    assert "https://arxiv.org/html/extracted/figures/fig1.png" in result
    assert "Figure 1: System overview" in result


def test_absolute_image_urls_unchanged() -> None:
    """Test that absolute image URLs are not modified."""
    html = """
    <figure>
        <img src="https://arxiv.org/html/2501.11120v1/assets/img.png" alt="Diagram"/>
        <figcaption>Figure 2: Results</figcaption>
    </figure>
    """

    result = convert_fragment_to_markdown(html, base_url="https://ar5iv.labs.arxiv.org/html/2501.11120v1")
    assert "https://arxiv.org/html/2501.11120v1/assets/img.png" in result


def test_latexml_dotted_figure_directory_is_restored_to_the_paper_origin() -> None:
    html = (
        '<figure><img src="https://Figures.Hand.Highlight/loss.png" alt="Loss">'
        "</figure>"
    )

    result = convert_fragment_to_markdown(
        html,
        base_url="https://arxiv.org/html/2411.04996",
    )

    assert result == (
        "![Loss](https://arxiv.org/html/2411.04996/"
        "Figures.Hand.Highlight/loss.png)"
    )


def test_no_base_url_preserves_relative_paths() -> None:
    """Test that without base_url, relative paths are preserved as-is."""
    html = """
    <figure>
        <img src="extracted/fig1.png" alt="Diagram"/>
    </figure>
    """

    result = convert_fragment_to_markdown(html)
    assert "extracted/fig1.png" in result


def test_latexml_equation_group_has_structured_displays_without_nested_dollars() -> None:
    html = """
    <table class="ltx_equationgroup" id="S2.EG1"><tbody id="S2.E1">
      <tr><td><math class="ltx_Math" alttext="fallback">
        <annotation encoding="application/x-tex">$$a &amp;= b + c$$</annotation>
      </math></td><td class="ltx_eqn_number">(3)</td></tr></tbody>
      <tbody id="S2.E2"><tr><td><math alttext="d = e"/></td><td class="ltx_tag">(4)</td></tr></tbody>
    </table>
    """
    result = convert_fragment_to_markdown(html)
    assert '<a id="S2.EG1"></a>' in result
    assert '<a id="S2.E1"></a>' in result
    assert '<a id="S2.E2"></a>' in result
    assert result.count("$$") == 4
    assert "$$a" not in result and "c$$" not in result
    assert "a &= b + c" in result and "d = e" in result
    assert "\\tag{3}" in result and "\\tag{4}" in result


def test_figure_preserves_anchor_caption_and_all_images_as_markdown() -> None:
    html = """
    <figure id="S1.F1"><img src="a.png" alt="First"><img src="b.png" alt="Second">
      <figcaption>Figure 1: Two panels.</figcaption></figure>
    """
    result = convert_fragment_to_markdown(html, base_url="https://ar5iv.labs.arxiv.org/html/1234.5678")
    assert result.startswith('<a id="S1.F1"></a>')
    assert "![First](https://ar5iv.labs.arxiv.org/html/a.png)" in result
    assert "![Second](https://ar5iv.labs.arxiv.org/html/b.png)" in result
    assert "*Figure 1: Two panels.*" in result


def test_flex_figure_preserves_rows_widths_panel_captions_and_anchors() -> None:
    html = """
    <figure id="S2.F1" class="ltx_figure">
      <div class="ltx_flex_figure">
        <div class="ltx_flex_cell ltx_flex_size_2">
          <figure id="S2.F1.sf1" class="ltx_figure ltx_figure_panel">
            <img src="first.png" alt="First panel">
            <figcaption>(a) Swiss Roll</figcaption>
          </figure>
        </div>
        <div class="ltx_flex_cell ltx_flex_size_2">
          <figure id="S2.F1.sf2" class="ltx_figure ltx_figure_panel">
            <img src="second.png" alt="Second panel">
            <figcaption>(b) Torus</figcaption>
          </figure>
        </div>
        <div class="ltx_flex_break"></div>
        <div class="ltx_flex_cell ltx_flex_size_1">
          <figure id="S2.F1.sf3" class="ltx_figure ltx_figure_panel">
            <img src="third.png" alt="Third panel">
          </figure>
        </div>
      </div>
      <figcaption>Figure 1: Manifold examples.</figcaption>
    </figure>
    """

    result = convert_fragment_to_markdown(html, base_url="https://arxiv.org/html/1234.5678v1")

    assert result == "\n".join(
        [
            '<PaperFigure id="S2.F1">',
            "",
            '<PaperFigureRow columns="2 2">',
            "",
            '<PaperFigurePanel id="S2.F1.sf1">',
            "",
            "![First panel](https://arxiv.org/html/first.png)",
            "",
            "*(a) Swiss Roll*",
            "",
            "</PaperFigurePanel>",
            "",
            '<PaperFigurePanel id="S2.F1.sf2">',
            "",
            "![Second panel](https://arxiv.org/html/second.png)",
            "",
            "*(b) Torus*",
            "",
            "</PaperFigurePanel>",
            "",
            "</PaperFigureRow>",
            "",
            '<PaperFigureRow columns="1">',
            "",
            '<PaperFigurePanel id="S2.F1.sf3">',
            "",
            "![Third panel](https://arxiv.org/html/third.png)",
            "",
            "</PaperFigurePanel>",
            "",
            "</PaperFigureRow>",
            "",
            "*Figure 1: Manifold examples.*",
            "",
            "</PaperFigure>",
        ]
    )


def test_flex_figure_wraps_cells_when_their_latexml_widths_fill_a_row() -> None:
    cells = "".join(
        f'<div class="ltx_flex_cell ltx_flex_size_3"><figure class="ltx_figure_panel"><img src="{index}.png"></figure></div>'
        for index in range(6)
    )

    result = convert_fragment_to_markdown(f'<figure><div class="ltx_flex_figure">{cells}</div></figure>')

    assert result.count('<PaperFigureRow columns="3 3 3">') == 2
    assert result.count("<PaperFigurePanel>") == 6


def test_flex_figure_preserves_an_empty_panel_as_a_layout_placeholder() -> None:
    html = """
    <figure><div class="ltx_flex_figure">
      <div class="ltx_flex_cell ltx_flex_size_3"><figure id="placeholder" class="ltx_figure_panel"></figure></div>
      <div class="ltx_flex_cell ltx_flex_size_3"><figure class="ltx_figure_panel"><img src="middle.png"></figure></div>
      <div class="ltx_flex_cell ltx_flex_size_3"><figure class="ltx_figure_panel"><img src="right.png"></figure></div>
    </div></figure>
    """

    result = convert_fragment_to_markdown(html)

    assert '<PaperFigureRow columns="3 3 3">' in result
    assert '<PaperFigurePanel id="placeholder">\n</PaperFigurePanel>' in result
    assert result.count("<PaperFigurePanel") == 3
    assert result.index('id="placeholder"') < result.index("middle.png") < result.index("right.png")
