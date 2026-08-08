#!/usr/bin/env python3

import os
import re
import sys
import shutil
import yaml
from pathlib import Path
from bs4 import BeautifulSoup
import markdown
import requests
from urllib.parse import urlparse
from datetime import datetime

# --- CONFIGURATION ---
MD_DIR = "md"
DITA_DIR = "dita"
HTML_DIR = "html"
TOC_MD = "toc.md"
METADATA_FILE = "metadata.yaml"
LOG_FILE = "generate-dita.log"

# ---------------------------------------------------------------------
def log(message: str):
    """Write message to stdout and append to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(line + "\n")

def load_metadata(metadata_file: str = METADATA_FILE) -> dict:
    """Load metadata from YAML file."""
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        # Validate required fields
        if not metadata.get('title'):
            log(f"⚠️  Warning: 'title' not found in {metadata_file}, using default")
            metadata['title'] = "Documentation"

        log(f"✅ Loaded metadata from {metadata_file}")
        log(f"   Title: {metadata.get('title')}")
        if metadata.get('author'):
            log(f"   Author: {metadata.get('author')}")

        return metadata
    except FileNotFoundError:
        log(f"⚠️  Warning: {metadata_file} not found, using defaults")
        return {'title': 'Documentation', 'language': 'en'}
    except yaml.YAMLError as e:
        log(f"⚠️  Warning: Error parsing {metadata_file}: {e}")
        return {'title': 'Documentation', 'language': 'en'}

def normalize_path(path: str) -> str:
    """Normalize a file path for consistent comparison."""
    # Strip leading ./ and /
    path = path.lstrip("./").lstrip("/")
    if path.endswith(".html"):
        path = path[:-5] + ".md"
    return os.path.normpath(path)

def resolve_image_path(img_src: str, md_file_path: str) -> str:
    """
    Resolve image path to point to the html directory.

    Since images are already downloaded in the html directory by generate-html.py,
    we just need to create a relative path from the DITA output location to the html directory.

    Args:
        img_src: The image src from the markdown/HTML (relative path)
        md_file_path: Path to the source markdown file

    Returns:
        Relative path from DITA directory to the image in html directory
    """
    # Skip data URIs and external URLs
    if img_src.startswith(('data:', 'http://', 'https://')):
        return img_src

    # The img_src is a path relative to the MD file location
    # We need to resolve it to an absolute path, then make it relative to DITA_DIR

    try:
        md_path = Path(md_file_path)

        # Resolve the image path relative to the markdown file
        if md_path.is_absolute():
            img_abs_path = (md_path.parent / img_src).resolve()
        else:
            img_abs_path = (Path(md_file_path).parent / img_src).resolve()

        # Make it relative to the DITA directory
        relative_to_dita = os.path.relpath(img_abs_path, DITA_DIR)

        log(f"✓ Resolved image path: {img_src} -> {relative_to_dita}")
        return relative_to_dita

    except Exception as e:
        log(f"⚠️  Could not resolve image path {img_src}: {e}")
        return img_src

def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))

def generate_anchor_slug(text: str) -> str:
    """Generate an anchor slug from heading text (same as markdown auto-generates).

    Example: "13.1 Extension Points Design" -> "131-extension-points-design"
    """
    import re
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug

# Code lines longer than this wrap on a Kindle rather than scrolling — Amazon
# supports `overflow` only as `hidden`, so there is no scrollable region to put
# a wide code sample in. These are starting points to calibrate against Kindle
# Previewer at the default font size on a 6" profile, not settled numbers.
CODE_LINE_WARN = 56
CODE_LINE_MAX = 64
# A token wider than this cannot break, and is clipped rather than wrapped.
CODE_TOKEN_MAX = 58
CODE_LINE_REPORT = "code-line-report.txt"


def report_long_code_lines(md_dir: Path) -> int:
    """List every fenced-code line too long to fit a Kindle screen.

    Reports against the Markdown, not the generated DITA, because the Markdown
    is what gets edited. Never fails the build: this is a worklist, and the
    ASCII diagrams in it cannot be fixed by shortening a line anyway.

    Returns the number of lines over CODE_LINE_MAX.
    """
    findings = []
    unbreakable = []

    for md_file in sorted(md_dir.glob("*.md")):
        in_fence = False
        fence_marker = None

        for number, line in enumerate(md_file.read_text(encoding="utf-8").split("\n"), 1):
            stripped = line.lstrip()

            if not in_fence:
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    in_fence = True
                    fence_marker = stripped[:3]
                continue

            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = None
                continue

            if len(line) > CODE_LINE_WARN:
                findings.append((md_file.name, number, len(line), line))

            # An unbreakable run wider than a line is worse than a long line.
            # The stylesheet uses `word-break: normal`, because break-all split
            # words mid-token on Kindle even when the whole word would have
            # fitted on the next line. The cost is that a token too wide to fit
            # cannot break at all, and `overflow` is supported only as `hidden`,
            # so it is clipped and the text is lost.
            for token in line.split():
                if len(token) > CODE_TOKEN_MAX:
                    unbreakable.append((md_file.name, number, len(token), token))

    if unbreakable:
        log(f"✂️  {len(unbreakable)} unbreakable token(s) over {CODE_TOKEN_MAX} "
            f"characters — these clip rather than wrap:")
        for name, number, length, token in sorted(unbreakable, key=lambda x: -x[2]):
            log(f"   {name}:{number} ({length}) {token[:60]}")

    over_max = [f for f in findings if f[2] > CODE_LINE_MAX]

    if not findings:
        log(f"✅ No code lines exceed {CODE_LINE_WARN} characters")
        return 0

    by_file = {}
    for name, number, length, text in findings:
        by_file.setdefault(name, []).append((number, length, text))

    log(f"📏 Code line lengths: {len(over_max)} line(s) over {CODE_LINE_MAX}, "
        f"{len(findings) - len(over_max)} between {CODE_LINE_WARN + 1} and {CODE_LINE_MAX}")
    log(f"   Full list: {CODE_LINE_REPORT}")

    for name in sorted(by_file, key=lambda n: -max(x[1] for x in by_file[n])):
        entries = by_file[name]
        worst = max(x[1] for x in entries)
        count = sum(1 for x in entries if x[1] > CODE_LINE_MAX)
        if count:
            log(f"   {name}: {count} over {CODE_LINE_MAX}, longest {worst}")

    with open(CODE_LINE_REPORT, "w", encoding="utf-8") as report:
        report.write(
            f"Fenced-code lines longer than {CODE_LINE_WARN} characters.\n"
            f"Lines over {CODE_LINE_MAX} are marked OVER; those will wrap on a\n"
            f"6\" Kindle at the default font size. Kindle cannot scroll a code\n"
            f"block horizontally, so the fix is to shorten the source line.\n\n"
        )
        for name in sorted(by_file):
            report.write(f"{name}\n")
            for number, length, text in sorted(by_file[name]):
                flag = "OVER" if length > CODE_LINE_MAX else "    "
                report.write(f"  {flag} {number:>5}  {length:>3}  {text}\n")
            report.write("\n")

    return len(over_max)


# Fence languages Prism has no grammar for, mapped to the nearest one it does.
# An unknown language makes fox.jason.prismjs emit PRSM002W and skip the block,
# so it silently loses its highlighting in both the PDF and the EPUB.
PRISM_LANGUAGE_ALIASES = {
    "jsonc": "json",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "console": "bash",
    "yml": "yaml",
    "ts": "typescript",
    "js": "javascript",
    "py": "python",
    "text": None,   # plain text: no highlighting wanted
    "txt": None,
    "plain": None,
}

# Characters that only ever appear in a drawn diagram, never in source code.
_DIAGRAM_UNICODE = re.compile(r"[←-⇿─-╿]")

# A line made of nothing but connector glyphs and whitespace.
_CONNECTOR_ONLY = re.compile(r"^[\s|+\-<>^v/\\.'`~=*_]+$")

# A line that opens with a connector, as the branches of an ASCII tree do.
_CONNECTOR_START = re.compile(r"^\s*[|+`\\]")


def looks_like_diagram(code_text: str) -> bool:
    """Is this fenced block a drawn diagram rather than source code?

    Diagrams get no `outputclass`, so Prism leaves them alone. Tokenizing one
    as bash — which the heuristics below happily did, since `|` and `-` are
    everywhere in an ASCII tree — colours arbitrary fragments of the drawing
    and produces highlighting that means nothing.
    """
    if not code_text or not code_text.strip():
        return False

    if _DIAGRAM_UNICODE.search(code_text):
        return True

    lines = [line for line in code_text.split("\n") if line.strip()]
    if len(lines) < 3:
        return False

    connector_only = sum(1 for line in lines if _CONNECTOR_ONLY.match(line))
    structural = sum(
        1 for line in lines
        if _CONNECTOR_ONLY.match(line) or _CONNECTOR_START.match(line)
        or "-->" in line or "<--" in line
    )

    # At least one pure connector line, and a third of the block structural.
    return connector_only >= 1 and structural >= max(2, len(lines) // 3)


def resolve_code_outputclass(explicit: str, code_text: str) -> str:
    """Settle the @outputclass for one fenced block, or None for neither.

    Returns `language-x` for source code, `diagram` for a drawing, None when
    the block is neither.

    Diagrams are marked rather than merely left unhighlighted. A code sample
    that wraps is still readable — that is what the hanging indent in ebook.css
    is for — but a wrapped diagram is destroyed, because its meaning is in the
    column alignment. Marking them lets the stylesheet set them at a size that
    fits the screen instead, which is the only way to keep a 78-column drawing
    intact on a 6" Kindle.
    """
    if explicit:
        normalized = explicit.strip().lower()
        if normalized in PRISM_LANGUAGE_ALIASES:
            mapped = PRISM_LANGUAGE_ALIASES[normalized]
            if mapped != normalized:
                log(f"🔤 Fence language '{normalized}' -> "
                    f"{mapped or 'none (plain text)'}")
            if not mapped:
                return "diagram" if looks_like_diagram(code_text) else None
            return f"language-{mapped}"
        return f"language-{normalized}"

    if looks_like_diagram(code_text):
        return "diagram"

    detected = detect_language(code_text)
    return f"language-{detected}" if detected else None


def detect_language(code_text: str) -> str:
    """
    Auto-detect programming language for unlabeled code blocks.

    Uses heuristics to identify common languages based on syntax patterns.
    Returns the detected language name or None if unable to detect.
    """
    if not code_text or not code_text.strip():
        return None

    code_lower = code_text.lower()
    code_stripped = code_text.strip()

    # JSON detection - starts with { or [ (common for JSON)
    if code_stripped.startswith(('{', '[')):
        # Check for JSON-like patterns
        if any(pattern in code_text for pattern in ['":', '": ', '":"']):
            return 'json'

    # TypeScript/JavaScript detection
    ts_js_patterns = [
        'interface ', 'type ', 'const ', 'let ', 'var ',
        'function ', '=>', 'async ', 'await ', 'import ',
        'export ', 'class ', 'extends ', 'implements '
    ]
    if any(pattern in code_text for pattern in ts_js_patterns):
        # TypeScript-specific patterns
        if any(ts_pattern in code_text for ts_pattern in [
            'interface ', 'type ', ': string', ': number', ': boolean',
            '<T>', 'extends ', 'implements '
        ]):
            return 'typescript'
        return 'javascript'

    # Bash/shell script detection
    bash_patterns = [
        '#!/bin/bash', '#!/bin/sh', 'npm ', 'yarn ', 'npx ',
        'git ', 'cd ', 'ls ', 'mkdir ', 'chmod ', 'echo $',
        'export ', '${', 'if [', 'then', 'elif', 'fi'
    ]
    if code_stripped.startswith('#!/bin/') or code_stripped.startswith('$ '):
        return 'bash'
    if any(pattern in code_text for pattern in bash_patterns):
        return 'bash'

    # YAML detection
    yaml_patterns = ['---', 'apiVersion:', 'kind:', 'metadata:', 'spec:']
    if code_stripped.startswith('---') or any(pattern in code_text for pattern in yaml_patterns):
        # Make sure it's not markdown frontmatter
        lines = code_text.split('\n')
        if len(lines) > 1 and any(':' in line and not line.strip().startswith('#') for line in lines):
            return 'yaml'

    # Dockerfile detection
    if any(code_stripped.startswith(cmd) for cmd in ['FROM ', 'RUN ', 'COPY ', 'ADD ', 'ENV ', 'WORKDIR ']):
        return 'dockerfile'

    # Python detection
    python_patterns = ['def ', 'import ', 'from ', 'class ', 'if __name__', 'print(']
    if any(pattern in code_text for pattern in python_patterns):
        return 'python'

    # Markdown detection
    if code_stripped.startswith('#') and '\n## ' in code_text:
        return 'markdown'

    # Vim script detection
    if 'set ' in code_text and ('syntax ' in code_text or 'colorscheme ' in code_text):
        return 'vim'

    # Lua detection
    if 'local ' in code_text or 'function(' in code_text or 'end' in code_text:
        if 'require' in code_lower or 'return' in code_lower:
            return 'lua'

    # If unable to detect, return None
    return None

def is_inline_image(img_element) -> bool:
    """
    Determine if an image is inline (icon) or standalone (figure).

    Inline images are:
    - Wrapped in a span
    - Part of a block element that contains non-whitespace text content

    Standalone images are:
    - Direct child of a block element with no other non-whitespace content
    """
    parent = img_element.parent
    if not parent:
        return False

    # If wrapped in a span, it's inline
    if parent.name == 'span':
        return True

    # Check if the parent block has other non-whitespace content
    if parent.name in ['p', 'div', 'li', 'td', 'th']:
        # Get all text content excluding the img itself
        text_content = ""
        for child in parent.children:
            if child.name != 'img' and hasattr(child, 'get_text'):
                text_content += child.get_text()
            elif isinstance(child, str):
                text_content += child

        # If there's non-whitespace text, the image is inline
        if text_content.strip():
            return True

    return False

def convert_element_content(element) -> str:
    """Convert HTML element content to DITA, handling links."""
    result = []
    for child in element.children:
        if isinstance(child, str):
            result.append(escape_xml(child))
        elif child.name == "a":
            # Check if this is an internal DITA link
            if child.get("data-dita-href"):
                href = child["data-dita-href"]
                link_text = child.get_text(strip=True)
                result.append(f'<xref href="{href}" format="dita">{escape_xml(link_text)}</xref>')
            else:
                # External link or unresolved link
                href = child.get("href", "")
                link_text = child.get_text(strip=True)
                if href.startswith("http://") or href.startswith("https://"):
                    result.append(f'<xref href="{escape_xml(href)}" format="html" scope="external">{escape_xml(link_text)}</xref>')
                else:
                    # Just text for unresolved internal links
                    # Preserve surrounding whitespace when flattening links
                    prev_sibling = child.previous_sibling
                    next_sibling = child.next_sibling

                    # Check if we need to add a space before
                    if isinstance(prev_sibling, str):
                        if prev_sibling and not prev_sibling[-1].isspace():
                            link_text = " " + link_text
                    elif prev_sibling is not None:  # It's an element
                        link_text = " " + link_text

                    # Check if we need to add a space after
                    if isinstance(next_sibling, str):
                        if next_sibling and not next_sibling[0].isspace():
                            link_text = link_text + " "
                    elif next_sibling is not None:  # It's an element
                        link_text = link_text + " "

                    result.append(escape_xml(link_text))
        elif child.name == "code":
            result.append(f'<codeph>{escape_xml(child.get_text())}</codeph>')
        elif child.name == "strong" or child.name == "b":
            result.append(f'<b>{escape_xml(child.get_text())}</b>')
        elif child.name == "em" or child.name == "i":
            result.append(f'<i>{escape_xml(child.get_text())}</i>')
        elif child.name == "br":
            # Line break - add a space and newline for better formatting in DITA
            result.append(' ')
        elif child.name == "img":
            img_src = child.get("src", "")
            img_alt = child.get("alt", "")
            if img_src:
                # Determine if this is an inline icon or standalone figure
                if is_inline_image(child):
                    # Inline icon: height limited to line height
                    result.append(f'<image href="{escape_xml(img_src)}" placement="inline">')
                else:
                    # Standalone figure: full size, max 50% screen width
                    result.append(f'<image href="{escape_xml(img_src)}" placement="break" align="left">')
                if img_alt:
                    result.append(f'<alt>{escape_xml(img_alt)}</alt>')
                result.append('</image>')
        else:
            result.append(escape_xml(child.get_text()))
    return "".join(result)

def convert_list_item_content(li_element) -> list:
    """Convert list item content, handling both inline and block elements."""
    result = []

    for child in li_element.children:
        if isinstance(child, str):
            text = child.strip()
            if text:
                result.append(escape_xml(text))
        elif child.name == "p":
            # Paragraph in list item
            result.append('<p>' + convert_element_content(child) + '</p>')
        elif child.name == "pre":
            # Code block in list item
            code_elem = child.find("code")
            if code_elem:
                classes = code_elem.get("class", [])
                language = None

                if isinstance(classes, list):
                    for cls in classes:
                        if cls.startswith("language-"):
                            language = cls.replace("language-", "")
                            break
                elif isinstance(classes, str) and classes.startswith("language-"):
                    language = classes.replace("language-", "")

                code_text = code_elem.get_text()
                outputclass = resolve_code_outputclass(language, code_text)

                if outputclass:
                    result.append(f'<codeblock outputclass="{outputclass}">{escape_xml(code_text)}</codeblock>')
                else:
                    result.append(f'<codeblock>{escape_xml(code_text)}</codeblock>')
            else:
                code_text = child.get_text()
                outputclass = resolve_code_outputclass(None, code_text)
                if outputclass:
                    result.append(f'<codeblock outputclass="{outputclass}">{escape_xml(code_text)}</codeblock>')
                else:
                    result.append(f'<codeblock>{escape_xml(code_text)}</codeblock>')
        elif child.name in ["ul", "ol"]:
            # Nested list - handle recursively
            # Note: For simplicity, we'll inline the list content
            # More sophisticated handling could be added here
            pass
        elif child.name == "code":
            result.append(f'<codeph>{escape_xml(child.get_text())}</codeph>')
        elif child.name == "strong" or child.name == "b":
            result.append(f'<b>{escape_xml(child.get_text())}</b>')
        elif child.name == "em" or child.name == "i":
            result.append(f'<i>{escape_xml(child.get_text())}</i>')
        elif child.name == "a":
            # Handle links
            if child.get("data-dita-href"):
                href = child["data-dita-href"]
                link_text = child.get_text(strip=True)
                result.append(f'<xref href="{href}" format="dita">{escape_xml(link_text)}</xref>')
            else:
                href = child.get("href", "")
                link_text = child.get_text(strip=True)
                if href.startswith("http://") or href.startswith("https://"):
                    result.append(f'<xref href="{escape_xml(href)}" format="html" scope="external">{escape_xml(link_text)}</xref>')
                elif href.startswith("#"):
                    # Fragment-only anchor (internal page link) - keep as xref
                    # Keep the # for DITA - it indicates a same-topic reference
                    result.append(f'<xref href="{href}">{escape_xml(link_text)}</xref>')
                else:
                    # Preserve surrounding whitespace when flattening other links
                    prev_sibling = child.previous_sibling
                    next_sibling = child.next_sibling

                    # Check if we need to add a space before
                    if isinstance(prev_sibling, str):
                        if prev_sibling and not prev_sibling[-1].isspace():
                            link_text = " " + link_text
                    elif prev_sibling is not None:  # It's an element
                        link_text = " " + link_text

                    # Check if we need to add a space after
                    if isinstance(next_sibling, str):
                        if next_sibling and not next_sibling[0].isspace():
                            link_text = link_text + " "
                    elif next_sibling is not None:  # It's an element
                        link_text = link_text + " "

                    result.append(escape_xml(link_text))

    return result

# Rough advance per character, as a fraction of the font size. Code cells are
# set in Courier New, which advances 0.6em for every glyph; prose is Helvetica,
# where 0.5em is a fair average across mixed-case text.
_MONO_ADVANCE = 0.6
_PROSE_ADVANCE = 0.5

# Cell padding, in em of the table font. A flat per-column cost, so it belongs
# in the weight: leave it out and every column is allocated its text width and
# then has padding taken back out of it.
_CELL_PADDING = 1.5


def _cell_widths(cell):
    """(longest unbreakable run, whole-cell width) for one cell, in em.

    The two numbers answer different questions. A cell wraps at spaces, so its
    natural width is only what it would like; the longest run without a space
    is what it *needs*, because nothing can break an identifier like
    OPENCODE_PROVIDER_ANTHROPIC_API_KEY and a column narrower than that one run
    is a column the text runs out of.
    """
    longest = natural = 0.0

    for text in cell.strings:
        mono = any(parent.name == "code" for parent in text.parents)
        advance = _MONO_ADVANCE if mono else _PROSE_ADVANCE
        natural += len(text) * advance
        for run in str(text).split():
            longest = max(longest, len(run) * advance)

    return longest, natural


def estimate_colwidths(rows, num_cols):
    """Proportional column weights for a table's colspecs, or None.

    Without these every column takes an equal share whatever it holds, which is
    how a column of environment-variable names came to need 210pt of a 154pt
    third of the table and ran off the page.

    A column is weighted by the longest run it has to fit, plus cell padding,
    with a floor of a quarter of its natural width so a column of prose is not
    squeezed to nothing next to one holding a long identifier. The floor is
    deliberately low: prose wraps and an identifier does not, so where the two
    compete the identifier has to win. Cells that span columns are skipped,
    since their width belongs to no single column.
    """
    longest = [0.0] * num_cols
    natural = [0.0] * num_cols

    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        for index, cell in enumerate(cells):
            if index >= num_cols or cell.get("colspan"):
                continue
            run, whole = _cell_widths(cell)
            longest[index] = max(longest[index], run)
            natural[index] = max(natural[index], whole)

    weights = [max(longest[i], natural[i] / 4) + _CELL_PADDING
               for i in range(num_cols)]
    total = sum(weights)
    if not total:
        return None

    # Scaled to a thousand rather than to the smallest column: rounding to
    # small integers costs enough precision to push a column a point short of
    # the string it exists to hold.
    return [max(1, round(1000 * w / total)) for w in weights]


def convert_table_to_dita(table_element) -> str:
    """Convert HTML table to DITA table with formatting preserved."""
    dita_table = ['    <table>']

    # Find thead and tbody
    thead = table_element.find("thead")
    tbody = table_element.find("tbody")

    # Determine column count from first row
    num_cols = 1
    if thead:
        first_row = thead.find("tr")
        if first_row:
            num_cols = len(first_row.find_all(["th", "td"]))
    elif tbody:
        first_row = tbody.find("tr")
        if first_row:
            num_cols = len(first_row.find_all(["td", "th"]))

    dita_table.append(f'      <tgroup cols="{num_cols}">')

    # Add column specifications, sized to what each column has to hold.
    all_rows = []
    for section in (thead, tbody):
        if section:
            all_rows.extend(section.find_all("tr"))
    widths = estimate_colwidths(all_rows, num_cols)

    for i in range(num_cols):
        if widths:
            dita_table.append(
                '        <colspec colname="c{}" colnum="{}" colwidth="{}*"/>'.format(
                    i + 1, i + 1, widths[i]))
        else:
            dita_table.append('        <colspec colname="c{}" colnum="{}"/>'.format(i+1, i+1))

    if thead:
        dita_table.append('        <thead>')
        for tr in thead.find_all("tr"):
            dita_table.append('          <row>')
            for th in tr.find_all(["th", "td"]):
                # Preserve formatting in table cells
                cell_content = convert_element_content(th)

                # Handle colspan/rowspan
                colspan = th.get("colspan")
                rowspan = th.get("rowspan")

                entry_attrs = []
                if colspan:
                    entry_attrs.append(f'namest="c1" nameend="c{colspan}"')
                if rowspan:
                    entry_attrs.append(f'morerows="{int(rowspan)-1}"')

                if entry_attrs:
                    dita_table.append(f'            <entry {" ".join(entry_attrs)}>{cell_content}</entry>')
                else:
                    dita_table.append(f'            <entry>{cell_content}</entry>')
            dita_table.append('          </row>')
        dita_table.append('        </thead>')

    if tbody:
        dita_table.append('        <tbody>')
        for tr in tbody.find_all("tr"):
            dita_table.append('          <row>')
            for td in tr.find_all(["td", "th"]):
                # Preserve formatting in table cells
                cell_content = convert_element_content(td)

                # Handle colspan/rowspan
                colspan = td.get("colspan")
                rowspan = td.get("rowspan")

                entry_attrs = []
                if colspan:
                    entry_attrs.append(f'namest="c1" nameend="c{colspan}"')
                if rowspan:
                    entry_attrs.append(f'morerows="{int(rowspan)-1}"')

                if entry_attrs:
                    dita_table.append(f'            <entry {" ".join(entry_attrs)}>{cell_content}</entry>')
                else:
                    dita_table.append(f'            <entry>{cell_content}</entry>')
            dita_table.append('          </row>')
        dita_table.append('        </tbody>')

    dita_table.append('      </tgroup>')
    dita_table.append('    </table>')
    return "\n".join(dita_table)

def strip_emojis(text: str) -> str:
    """
    Remove emojis from text for PDF compatibility.

    Apache FOP (the PDF formatter) doesn't support emoji fonts properly,
    rendering them as placeholder characters (#). This function removes
    emojis to ensure clean PDF output.
    """
    import re

    # Comprehensive emoji pattern covering most emoji ranges
    # This includes:
    # - Emoticons
    # - Miscellaneous Symbols and Pictographs
    # - Supplemental Symbols and Pictographs
    # - Transport and Map Symbols
    # - Enclosed characters
    # - Various other symbol ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\u2600-\u26FF"          # Misc symbols (NOT including box-drawing U+2500-U+257F)
        "\u2700-\u27BF"          # Dingbats
        "\U0001f926-\U0001f937"  # Face palm, shrug, etc.
        "\u200d"                 # Zero width joiner
        "\ufe0f"                 # Variation selector
        "\u2640-\u2642"          # Gender symbols
        "\u23cf"                 # Eject symbol
        "\u23e9"                 # Fast forward
        "\u231a"                 # Watch
        "\u3030"                 # Wavy dash
        "\u20e3"                 # Combining Enclosing Keycap
        "]+",
        flags=re.UNICODE
    )

    # Remove emojis
    text = emoji_pattern.sub('', text)

    # Clean up spacing issues after emoji removal
    # Important: Skip cleanup inside code blocks to preserve indentation
    lines = text.split('\n')
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        # Check for code fence markers
        stripped = line.strip()
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue

        # Skip cleanup for lines inside code blocks
        if in_code_block:
            cleaned_lines.append(line)
            continue
        # Check if this is a list item
        stripped = line.lstrip()
        is_list = stripped.startswith(('- ', '* ', '+ ')) or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in ['. ', ') '])

        if is_list:
            # For list items, preserve the structure but clean up after the marker
            # Extract indent, marker, and content
            indent = line[:len(line) - len(stripped)]
            if stripped.startswith(('- ', '* ', '+ ')):
                marker = stripped[:2]
                content = stripped[2:]
            else:
                # Numbered list
                marker_end = stripped.find('. ')
                if marker_end == -1:
                    marker_end = stripped.find(') ')
                marker = stripped[:marker_end + 2]
                content = stripped[marker_end + 2:]

            # Clean up the content part
            # Fix multiple spaces
            content = re.sub(r'  +', ' ', content)


            cleaned_lines.append(indent + marker + content)
        else:
            # For non-list lines, clean up normally
            # Fix multiple spaces
            line = re.sub(r'  +', ' ', line)
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def fix_list_spacing(md_text: str) -> str:
    """
    Ensure blank lines before lists for proper markdown parsing.

    Markdown parsers require a blank line before list items to recognize them
    as block-level lists. This function adds blank lines where needed.
    """
    lines = md_text.split('\n')
    result = []

    for i, line in enumerate(lines):
        # Check if current line is a list item
        stripped = line.strip()
        is_list_item = stripped.startswith(('- ', '* ', '+ ')) or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in ['. ', ') '])

        if is_list_item and i > 0:
            prev_line = lines[i-1].strip()
            # Check if previous line is NOT a list item and NOT blank
            prev_is_list = prev_line.startswith(('- ', '* ', '+ ')) or (len(prev_line) > 2 and prev_line[0].isdigit() and prev_line[1:3] in ['. ', ') '])

            if prev_line and not prev_is_list:
                # Need blank line before this list
                result.append('')

        result.append(line)

    return '\n'.join(result)

def preprocess_markdown_tables(md_text: str) -> str:
    """
    Preprocess markdown to handle tables with multi-row headers.

    Detects tables with multiple header rows before the separator and converts
    them to HTML tables that will be properly parsed and converted to DITA.
    """
    lines = md_text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        # Check if we're at the start of a table
        if i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
            # Extract the complete table
            table_lines = []
            j = i

            while j < len(lines):
                line = lines[j].strip()
                if line.startswith('|') and line.endswith('|'):
                    table_lines.append(lines[j])
                    j += 1
                elif line == '' and j + 1 < len(lines) and lines[j + 1].strip().startswith('|'):
                    # Empty line within table - skip it but continue
                    j += 1
                else:
                    # End of table
                    break

            # Check if this table has a multi-row header
            separator_index = -1
            for k, line in enumerate(table_lines):
                if _is_separator_row(line):
                    separator_index = k
                    break

            # If we found a separator and there are multiple header rows
            if separator_index > 1:
                log(f"🔍 Found table with {separator_index} header rows, normalizing to standard format")
                normalized_table = _normalize_multirow_header_table(table_lines, separator_index)
                result.append(normalized_table)
            else:
                # Standard table - keep as-is
                result.extend(table_lines)

            i = j
        else:
            result.append(lines[i])
            i += 1

    return '\n'.join(result)

def _is_separator_row(line: str) -> bool:
    """Check if a line is a table separator row (contains dashes between pipes)."""
    stripped = line.strip()
    if not stripped.startswith('|') or not stripped.endswith('|'):
        return False

    # Remove leading and trailing pipes
    content = stripped[1:-1]
    parts = content.split('|')

    # Check if all parts are separator cells (dashes, colons, spaces)
    has_separator = False
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Should only contain -, :, and spaces
        if not all(c in '-: ' for c in part):
            return False
        # Should have at least one dash
        if '-' in part:
            has_separator = True

    return has_separator

def _parse_table_row(row: str) -> list:
    """Parse a table row into individual cells."""
    # Remove leading and trailing pipes and whitespace
    content = row.strip()
    if content.startswith('|'):
        content = content[1:]
    if content.endswith('|'):
        content = content[:-1]

    # Split by pipe and return cells
    cells = [cell.strip() for cell in content.split('|')]
    return cells

def _normalize_multirow_header_table(table_lines: list, separator_index: int) -> str:
    """
    Normalize a markdown table with multi-row headers to HTML format.

    This function converts tables with multiple header rows into HTML tables
    that preserve the multi-row header structure using rowspan or multiple <tr> rows.

    Args:
        table_lines: All lines of the table
        separator_index: Index of the separator row

    Returns:
        HTML table string
    """
    header_rows = table_lines[:separator_index]
    data_rows = table_lines[separator_index + 1:]  # Skip separator

    # Parse all rows
    parsed_headers = [_parse_table_row(row) for row in header_rows]
    parsed_data = [_parse_table_row(row) for row in data_rows]

    # Build HTML table
    html_parts = ['<table>']
    html_parts.append('  <thead>')

    # Add all header rows as separate <tr> elements
    for header_row in parsed_headers:
        html_parts.append('    <tr>')
        for cell in header_row:
            # Preserve bold formatting if present
            cell_html = cell.replace('**', '')  # Remove markdown bold for now
            if cell.startswith('**') and cell.endswith('**'):
                html_parts.append(f'      <th><b>{cell_html}</b></th>')
            else:
                html_parts.append(f'      <th>{cell_html}</th>')
        html_parts.append('    </tr>')

    html_parts.append('  </thead>')
    html_parts.append('  <tbody>')

    # Add data rows
    for data_row in parsed_data:
        html_parts.append('    <tr>')
        for cell in data_row:
            # Preserve bold formatting if present
            if cell.startswith('**') and cell.endswith('**'):
                cell_html = cell[2:-2]  # Remove ** markers
                html_parts.append(f'      <td><b>{cell_html}</b></td>')
            else:
                html_parts.append(f'      <td>{cell}</td>')
        html_parts.append('    </tr>')

    html_parts.append('  </tbody>')
    html_parts.append('</table>')

    return '\n'.join(html_parts)

def md_to_dita_topic(md_path: str, topic_id: str, title: str, file_to_topic: dict = None) -> str:
    """Convert a Markdown file to a DITA topic."""
    if not os.path.exists(md_path):
        log(f"⚠️  Missing Markdown file: {md_path}")
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">
<topic id="{topic_id}">
  <title>{title}</title>
  <body>
    <p><i>Missing file: {md_path}</i></p>
  </body>
</topic>
"""

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Strip emojis for PDF compatibility (Apache FOP doesn't support emoji fonts)
    md_content = strip_emojis(md_content)

    # Preprocess markdown to fix list spacing (must come before other preprocessing)
    md_content = fix_list_spacing(md_content)

    # Preprocess markdown to handle tables with multi-row headers
    md_content = preprocess_markdown_tables(md_content)

    # Convert Markdown to HTML first
    html = markdown.markdown(md_content, extensions=[
        "extra",
        "tables",
        "fenced_code",
        "codehilite",  # Better code block handling with language detection
        "nl2br"        # Preserve line breaks in code examples
    ])
    soup = BeautifulSoup(html, "html.parser")

    # Process internal links if we have the mapping
    if file_to_topic:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip fragment-only anchors
            if href.startswith("#"):
                continue

            # Extract base path from href (handle both absolute and relative URLs)
            # Examples:
            #   /photoshop/using/blending-modes.html#new-mode → /photoshop/using/blending-modes
            #   https://helpx.adobe.com/photoshop/using/blending-modes.html#new-mode → /photoshop/using/blending-modes
            #   ../layers.html → ../layers

            base_path = href

            # If it's an absolute URL, extract the path
            if href.startswith("http://") or href.startswith("https://"):
                parsed = urlparse(href)
                # Only process URLs from the same domain (helpx.adobe.com)
                if "helpx.adobe.com" in parsed.netloc or "adobe.com" in parsed.netloc:
                    base_path = parsed.path
                else:
                    # Different domain - keep as external link
                    continue

            # Strip anchor/fragment and query parameters
            if '#' in base_path:
                base_path = base_path.split('#')[0]
            if '?' in base_path:
                base_path = base_path.split('?')[0]

            # Strip file extension (.html)
            if base_path.endswith('.html'):
                base_path = base_path[:-5]

            # Skip if path is empty after stripping
            if not base_path or base_path == '/':
                continue

            # Normalize the link path to check against our file mapping
            # Add .md extension and MD_DIR prefix for comparison
            # The file mapping uses paths like "md/photoshop/using/file.md"
            link_path = base_path + '.md'
            # Convert to format matching file_to_topic keys (e.g., "md/photoshop/using/file.md")
            if link_path.startswith('/'):
                link_path = '.' + link_path
            elif not link_path.startswith('./'):
                link_path = './' + link_path

            # Join with MD_DIR to match the format in file_to_topic
            link_path_with_dir = os.path.join(MD_DIR, link_path)
            normalized = normalize_path(link_path_with_dir)

            # Check if this file is in our DITA map
            if normalized in file_to_topic:
                target_topic = file_to_topic[normalized]
                # Mark this as an internal link for DITA conversion
                a_tag["data-dita-href"] = target_topic["filename"]
                a_tag["data-dita-scope"] = "local"
                log(f"🔗 Found internal link: {href} → {target_topic['filename']}")
            else:
                # File doesn't exist in our collection - keep as external link
                log(f"⚠️  Link to file not in collection (will remain external): {base_path}")

    # Process images - update paths to point to html directory
    for img_tag in soup.find_all("img"):
        img_src = img_tag.get("src", "")
        if img_src:
            # Skip data URIs (embedded images) - they cause issues in DITA
            if img_src.startswith("data:"):
                log(f"⏭️  Skipping data URI image (embedded content)")
                img_tag.decompose()  # Remove the image tag entirely
                continue

            # Resolve image path to point to html directory
            resolved_path = resolve_image_path(img_src, md_path)

            # If it's an external URL, skip it
            if resolved_path.startswith(("http://", "https://")):
                log(f"⏭️  Skipping external image URL: {resolved_path}")
                img_tag.decompose()
            else:
                img_tag["src"] = resolved_path

    # Start building DITA topic
    dita_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">',
        f'<topic id="{topic_id}">',
        f'  <title>{title}</title>',
        '  <body>'
    ]

    # Convert HTML elements to DITA
    for element in soup.children:
        if element.name == "h1":
            # Skip h1 since we use title
            continue
        elif element.name in ["h2", "h3", "h4", "h5", "h6"]:
            # Convert headings to sections with anchor IDs
            # Use separator=' ' to preserve word boundaries around inline
            # elements like <code> (get_text(strip=True) would collapse
            # "The <code>pyproject.toml</code> Dep..." into
            # "Thepyproject.tomlDep...")
            import re
            plain_text = re.sub(r'\s+', ' ', element.get_text(separator=' ')).strip()
            # Generate anchor slug for cross-references
            anchor_id = generate_anchor_slug(plain_text)
            # Build rich title content preserving inline formatting (codeph, b, i)
            title_content = convert_element_content(element)
            # Add anchor element before section for cross-references
            # Using <ph> (phrase) element with id - DITA-OT will convert this to <span id="..."> in HTML
            dita_parts.append(f'    <p><ph id="{anchor_id}"/></p>')
            dita_parts.append(f'    <section><title>{title_content}</title></section>')
        elif element.name == "p":
            # Handle paragraphs with potential links
            dita_parts.append('    <p>' + convert_element_content(element) + '</p>')
        elif element.name == "blockquote":
            # Handle blockquotes (convert to note/lq element)
            # Extract all paragraphs from blockquote
            for p in element.find_all("p", recursive=False):
                content = convert_element_content(p)
                dita_parts.append(f'    <lq>{content}</lq>')
        elif element.name == "ul":
            dita_parts.append('    <ul>')
            for li in element.find_all("li", recursive=False):
                li_parts = convert_list_item_content(li)
                dita_parts.append('      <li>')
                dita_parts.extend(['        ' + part for part in li_parts])
                dita_parts.append('      </li>')
            dita_parts.append('    </ul>')
        elif element.name == "ol":
            dita_parts.append('    <ol>')
            for li in element.find_all("li", recursive=False):
                li_parts = convert_list_item_content(li)
                dita_parts.append('      <li>')
                dita_parts.extend(['        ' + part for part in li_parts])
                dita_parts.append('      </li>')
            dita_parts.append('    </ol>')
        elif element.name == "pre":
            # Extract code element and language class
            code_elem = element.find("code")
            if code_elem:
                # Check for language class (e.g., "language-typescript")
                classes = code_elem.get("class", [])
                language = None

                if isinstance(classes, list):
                    for cls in classes:
                        if cls.startswith("language-"):
                            language = cls.replace("language-", "")
                            break
                elif isinstance(classes, str) and classes.startswith("language-"):
                    language = classes.replace("language-", "")

                code_text = code_elem.get_text()
                outputclass = resolve_code_outputclass(language, code_text)

                # Add outputclass for syntax highlighting, or to mark a diagram
                if outputclass:
                    dita_parts.append(f'    <codeblock outputclass="{outputclass}">{escape_xml(code_text)}</codeblock>')
                else:
                    dita_parts.append(f'    <codeblock>{escape_xml(code_text)}</codeblock>')
            else:
                # Fallback to pre content
                code_text = element.get_text()
                outputclass = resolve_code_outputclass(None, code_text)
                if outputclass:
                    dita_parts.append(f'    <codeblock outputclass="{outputclass}">{escape_xml(code_text)}</codeblock>')
                else:
                    dita_parts.append(f'    <codeblock>{escape_xml(code_text)}</codeblock>')
        elif element.name == "table":
            dita_parts.append(convert_table_to_dita(element))
        elif element.name == "img":
            img_src = element.get("src", "")
            img_alt = element.get("alt", "")
            if img_src:
                # Block-level images are standalone figures
                # placement="break" allows the image to be full size but max 50% screen width
                dita_parts.append(f'    <image href="{escape_xml(img_src)}" placement="break" align="left">')
                if img_alt:
                    dita_parts.append(f'      <alt>{escape_xml(img_alt)}</alt>')
                dita_parts.append('    </image>')

    dita_parts.append('  </body>')
    dita_parts.append('</topic>')

    return _deduplicate_anchor_ids("\n".join(dita_parts))


def _deduplicate_anchor_ids(topic_xml: str) -> str:
    """Make every @id unique within one topic.

    Two headings that slugify the same — "Availability" appearing in more than
    one section, say — produce the same anchor id, which DITA-OT reports as
    DOTJ057E and which leaves the second anchor unreachable. Later occurrences
    are suffixed. Cross-references keep resolving because they target the slug,
    which still belongs to the first occurrence.
    """
    seen = {}

    def rename(match):
        anchor = match.group(1)
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        if not count:
            return match.group(0)
        log(f"   ⚠️  Duplicate anchor '{anchor}' renamed to '{anchor}-{count + 1}'")
        return f'<ph id="{anchor}-{count + 1}"/>'

    return re.sub(r'<ph id="([^"]+)"/>', rename, topic_xml)

def convert_headings_to_lists(md_text: str) -> str:
    """Convert heading-based TOC to nested list format.

    Example input:
        # Title
        ## Section 1
        1. [Link 1](url1)
        2. [Link 2](url2)
        ### Subsection 1.1
        1. [Link 3](url3)
        ## Section 2
        1. [Link 4](url4)

    Example output:
        1. Section 1
            1. [Link 1](url1)
            2. [Link 2](url2)
            3. Subsection 1.1
                1. [Link 3](url3)
        2. Section 2
            1. [Link 4](url4)
    """
    import re
    lines = md_text.split('\n')
    output = []
    heading_stack = []  # Stack to track heading levels (actual # count)
    item_counters = {}  # Counter for each indent level

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a heading
        if line.strip().startswith('#'):
            # Parse heading level
            heading_match = line.strip()
            level = 0
            while heading_match.startswith('#'):
                level += 1
                heading_match = heading_match[1:]
            heading_text = heading_match.strip()

            # Skip the main title (level 1)
            if level == 1:
                i += 1
                continue

            # Adjust for ## being level 0 in our indentation
            indent_level = level - 2

            # Pop stack if we're going back to a higher level (fewer #s)
            while heading_stack and heading_stack[-1] >= level:
                heading_stack.pop()

            # Push current level
            heading_stack.append(level)

            # Clear counters for deeper levels
            for clear_level in range(indent_level + 1, 10):
                if clear_level in item_counters:
                    del item_counters[clear_level]

            # Increment counter at current level (for tracking child items)
            item_counters[indent_level] = item_counters.get(indent_level, 0) + 1

            # Add the heading as a list item (no link)
            # Strip any leading numbers from the heading text (like "1. " or "2.1 ")
            # to avoid double-numbering issues with markdown parsing
            clean_heading = heading_text
            # Check if starts with number(s) followed by dot and space
            match = re.match(r'^[\d.]+\s+(.+)$', heading_text)
            if match:
                clean_heading = match.group(1)

            # For level 0 (##), use no indent and add numbering
            # For deeper levels (###), use proper indentation
            if indent_level == 0:
                # Top-level heading - use simple numbered list
                output.append(f"{item_counters[indent_level]}. {clean_heading}")
            else:
                # Nested heading - use 4 spaces per level (CommonMark requires 4 spaces for nesting)
                indent = '    ' * indent_level
                output.append(f"{indent}{item_counters[indent_level]}. {clean_heading}")

        elif line.strip() and not line.strip().startswith('#'):
            # This is content under a heading (likely a numbered or bullet list)
            stripped = line.strip()

            # Check if it's a numbered list item with a link
            if stripped and stripped[0].isdigit() and '. ' in stripped[:5]:
                # This is a numbered list item - extract the content after "N. "
                content = stripped.split('. ', 1)[1] if '. ' in stripped else stripped

                # Indent under the current heading
                indent_level = len(heading_stack)
                # Use 4 spaces per level (CommonMark requires 4 spaces for nesting)
                indent = '    ' * indent_level

                # Increment counter for items at this level
                item_counters[indent_level] = item_counters.get(indent_level, 0) + 1

                output.append(f"{indent}{item_counters[indent_level]}. {content}")
            elif stripped.startswith('- '):
                # This is a bullet list item - extract the content after "- "
                content = stripped[2:].strip()

                # Indent under the current heading
                indent_level = len(heading_stack)
                # Use 4 spaces per level (CommonMark requires 4 spaces for nesting)
                indent = '    ' * indent_level

                # Increment counter for items at this level
                item_counters[indent_level] = item_counters.get(indent_level, 0) + 1

                output.append(f"{indent}{item_counters[indent_level]}. {content}")

        i += 1

    return '\n'.join(output)

def create_preface_topic(dita_dir: Path, metadata: dict) -> str:
    """
    Create a preface topic with copyright information from metadata.
    Returns the filename of the created preface topic, or None if no metadata.
    """
    if not metadata:
        return None

    # Check if we have any relevant metadata to display
    has_copyright_info = any([
        metadata.get('author'),
        metadata.get('publisher'),
        metadata.get('date'),
        metadata.get('rights'),
        metadata.get('isbn')
    ])

    if not has_copyright_info:
        return None

    log("📄 Creating preface topic with copyright information...")

    preface_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE topic PUBLIC "-//OASIS//DTD DITA Topic//EN" "topic.dtd">',
        '<topic id="preface_copyright">',
        '  <title>Copyright Information</title>',
        '  <body>'
    ]

    # Copyright notice
    if metadata.get('rights'):
        preface_parts.append('    <p>')
        preface_parts.append(f'      {escape_xml(metadata["rights"])}')
        preface_parts.append('    </p>')

    # Author
    if metadata.get('author'):
        authors = metadata['author']
        if isinstance(authors, list):
            preface_parts.append('    <p>')
            preface_parts.append('      <b>Author(s):</b> ')
            preface_parts.append(escape_xml(', '.join(authors)))
            preface_parts.append('    </p>')
        else:
            preface_parts.append('    <p>')
            preface_parts.append(f'      <b>Author:</b> {escape_xml(authors)}')
            preface_parts.append('    </p>')

    # Publisher
    if metadata.get('publisher'):
        preface_parts.append('    <p>')
        preface_parts.append(f'      <b>Publisher:</b> {escape_xml(metadata["publisher"])}')
        preface_parts.append('    </p>')

    # Publication date
    if metadata.get('date'):
        preface_parts.append('    <p>')
        preface_parts.append(f'      <b>Publication Date:</b> {escape_xml(metadata["date"])}')
        preface_parts.append('    </p>')

    # ISBN
    if metadata.get('isbn'):
        preface_parts.append('    <p>')
        preface_parts.append(f'      <b>ISBN:</b> {escape_xml(metadata["isbn"])}')
        preface_parts.append('    </p>')

    preface_parts.append('  </body>')
    preface_parts.append('</topic>')

    # Write preface topic
    preface_filename = "preface_copyright.dita"
    preface_path = dita_dir / preface_filename
    with open(preface_path, "w", encoding="utf-8") as f:
        f.write("\n".join(preface_parts))

    log(f"✅ Generated preface topic: {preface_filename}")
    return preface_filename

def parse_markdown_toc(md_file: str) -> tuple:
    """Convert toc.md to HTML and parse with BeautifulSoup.

    Supports two TOC formats:
    1. Numbered list format (indented with spaces/tabs)
    2. Heading-based format (using # ## ### for hierarchy)

    Returns: (BeautifulSoup object, extracted_title or None)
    """
    if not os.path.exists(md_file):
        raise FileNotFoundError(f"{md_file} not found")

    with open(md_file, "r", encoding="utf-8") as f:
        toc_md_text = f.read()

    # Extract title from first # heading if present
    extracted_title = None
    for line in toc_md_text.split('\n'):
        if line.strip().startswith('# '):
            extracted_title = line.strip()[2:].strip()
            log(f"📖 Extracted title from TOC: {extracted_title}")
            break

    # DON'T convert heading-based TOCs to lists - preserve original structure
    # The user wants the TOC.md structure to be preserved exactly
    # We'll extract links from headings, paragraphs, and lists directly

    # Preprocess markdown to fix nested list parsing issues
    # The markdown library has trouble with nested lists that have blank lines
    # We need to remove blank lines within list structures
    lines = toc_md_text.split('\n')
    cleaned_lines = []

    for i, line in enumerate(lines):
        # Skip blank lines that are between indented list items
        if line.strip() == '':
            # Check if we're in the middle of a list (previous and next lines are indented)
            prev_is_list = i > 0 and lines[i-1].strip() and (lines[i-1].startswith('  ') or lines[i-1].startswith('\t'))
            next_is_list = i < len(lines)-1 and lines[i+1].strip() and (lines[i+1].startswith('  ') or lines[i+1].startswith('\t'))

            if prev_is_list and next_is_list:
                # Skip this blank line - it breaks nested list parsing
                continue

        cleaned_lines.append(line)

    cleaned_md = '\n'.join(cleaned_lines)

    toc_html = markdown.markdown(cleaned_md, extensions=["extra"])
    soup = BeautifulSoup(toc_html, "html.parser")
    return soup, extracted_title

def build_dita_map(toc_soup: BeautifulSoup, dita_dir: Path, base_path: str = "", title: str = None, metadata: dict = None) -> tuple:
    """Build DITA bookmap from TOC and generate topic files. Returns (bookmap_content, title)."""

    # Use provided title, or extract from metadata, or extract from first link in TOC, or use default
    doc_title = title
    if not doc_title and metadata:
        doc_title = metadata.get('title')
    if not doc_title:
        # Try to extract from first link
        top_ol = toc_soup.find(["ol", "ul"])
        if top_ol:
            first_li = top_ol.find("li")
            if first_li:
                first_link = first_li.find("a")
                if first_link:
                    doc_title = first_link.get_text(strip=True)

        # Final fallback
        if not doc_title:
            doc_title = "Documentation"

    log(f"📖 Document title: {doc_title}")

    # First pass: collect all files (build mapping first)
    topic_counter = [0]
    file_to_topic = {}
    files_to_process = []

    def collect_links_from_element(elem, elem_type="element"):
        """Collect all markdown links from an element (list item or paragraph)."""
        # Get ALL links from this element (paragraphs can have multiple links)
        link_tags = elem.find_all("a", href=True, recursive=False)
        if not link_tags:
            return

        log(f"  🔍 Processing {elem_type}: found {len(link_tags)} link(s)")

        for link_tag in link_tags:
            title = link_tag.get_text(strip=True)
            href = link_tag["href"]

            log(f"    → Link: '{title}' -> {href}")

            # Skip URLs
            if href.startswith("http://") or href.startswith("https://"):
                log(f"      ⏭️  Skipping external URL")
                continue

            # Normalize path
            if href.startswith("/"):
                href = "." + href
            elif not href.startswith("./"):
                href = "./" + href

            href = os.path.join(base_path, href)
            if href.endswith(".html"):
                href = href[:-5] + ".md"

            # Build mapping
            normalized = normalize_path(href)
            if normalized not in file_to_topic:
                topic_counter[0] += 1
                topic_id = f"topic_{topic_counter[0]}"
                topic_filename = f"{topic_id}.dita"

                file_to_topic[normalized] = {
                    "id": topic_id,
                    "filename": topic_filename,
                    "title": title
                }
                files_to_process.append((href, topic_id, title, topic_filename))
                log(f"      ✅ Added as {topic_id}")
            else:
                log(f"      ⏭️  Already in mapping")

    def collect_files(ol, level=1):
        """Collect files from list structures."""
        for li in ol.find_all("li", recursive=False):
            collect_links_from_element(li, elem_type=f"list item (level {level})")
            nested_ol = li.find(["ol", "ul"], recursive=False)
            if nested_ol:
                collect_files(nested_ol, level + 1)

    # Collect from document in depth-first order (h2 followed by its list)
    log("📑 Collecting links from document in depth-first order...")

    headings = toc_soup.find_all("h2")
    log(f"  Found {len(headings)} h2 heading(s)")

    for i, h2 in enumerate(headings):
        # Collect links from the h2 (Part or README)
        collect_links_from_element(h2, elem_type=f"h2 heading {i+1}")

        # Find the next list (if any) before the next h2
        next_elem = h2.find_next_sibling()
        while next_elem:
            # Skip text nodes and NavigableStrings
            if not hasattr(next_elem, 'name'):
                next_elem = next_elem.find_next_sibling()
                continue

            # If we hit another h2, stop looking (no list for this h2)
            if next_elem.name == "h2":
                break

            # If we find a list, collect it and stop
            if next_elem.name in ["ol", "ul"]:
                log(f"  📋 Processing list after h2 heading {i+1}")
                collect_files(next_elem)
                break

            # Skip hr and other elements, keep looking
            next_elem = next_elem.find_next_sibling()

    log(f"📋 Found {len(file_to_topic)} files to convert to DITA")

    # Second pass: generate topics with link resolution
    for href, topic_id, title, topic_filename in files_to_process:
        # Convert MD to DITA with file mapping for link resolution
        dita_content = md_to_dita_topic(href, topic_id, title, file_to_topic)

        # Write DITA file
        topic_path = dita_dir / topic_filename
        with open(topic_path, "w", encoding="utf-8") as f:
            f.write(dita_content)

        log(f"✅ Generated DITA topic: {topic_filename} for {href}")

    # Save file-to-topic mapping for EPUB generation
    # This allows EPUB to use original markdown filenames for HTML files
    mapping_file = dita_dir / "file_to_topic.json"
    import json
    # Convert mapping to save original filenames
    simplified_mapping = {}
    for file_path, topic_info in file_to_topic.items():
        # Extract just the filename without path for cleaner HTML names
        # e.g., "./md/chapter-01-introduction.md" -> "chapter-01-introduction"
        base_name = Path(file_path).stem
        simplified_mapping[topic_info['filename']] = {
            'original_name': base_name,
            'topic_id': topic_info['id']
        }
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_mapping, f, indent=2)
    log(f"💾 Saved file-to-topic mapping: {mapping_file.name}")

    # Third pass: build DITA bookmap with metadata
    bookmap_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE bookmap PUBLIC "-//OASIS//DTD DITA BookMap//EN" "bookmap.dtd">',
        '<bookmap>',
        f'  <booktitle>',
        f'    <mainbooktitle>{escape_xml(doc_title)}</mainbooktitle>'
    ]

    # Add subtitle if present.
    # <booktitle> content model is (booklibrary?, mainbooktitle, booktitlealt*) —
    # the alternate title is a sibling <booktitlealt>, not a nested <booktitle>.
    if metadata and metadata.get('subtitle'):
        bookmap_parts.append(f'    <booktitlealt>{escape_xml(metadata["subtitle"])}</booktitlealt>')

    bookmap_parts.append(f'  </booktitle>')

    # Add metadata section.
    # <bookmeta> is an ordered content model: (author|authorinformation)*,
    # then publisherinformation*, then bookrights*. Emitting these out of order,
    # or putting <personname>/<organizationname> directly inside <author>/<publisher>,
    # is rejected by the bookmap DTD.
    if metadata:
        bookmap_parts.append('  <bookmeta>')

        # Add author information
        if metadata.get('author'):
            authors = metadata['author']
            if isinstance(authors, str):
                authors = [authors]
            for author in authors:
                bookmap_parts.append(f'    <author>{escape_xml(author)}</author>')

        # Add publisher
        if metadata.get('publisher'):
            bookmap_parts.append('    <publisherinformation>')
            bookmap_parts.append(f'      <organization>{escape_xml(metadata["publisher"])}</organization>')
            bookmap_parts.append('    </publisherinformation>')

        # Add copyright/rights
        if metadata.get('rights'):
            bookmap_parts.append('    <bookrights>')
            bookmap_parts.append(f'      <copyrfirst>')
            bookmap_parts.append(f'        <year>{datetime.now().year}</year>')
            bookmap_parts.append(f'      </copyrfirst>')
            bookmap_parts.append(f'      <bookowner>')
            if metadata.get('author'):
                author = metadata['author'] if isinstance(metadata['author'], str) else metadata['author'][0]
                bookmap_parts.append(f'        <person>{escape_xml(author)}</person>')
            bookmap_parts.append(f'      </bookowner>')
            bookmap_parts.append('    </bookrights>')

        bookmap_parts.append('  </bookmeta>')

    log(f"✅ Generated bookmap with metadata")

    # Create preface topic with copyright information
    preface_filename = create_preface_topic(dita_dir, metadata)

    # Add frontmatter section with preface if it was created
    if preface_filename:
        bookmap_parts.append('  <frontmatter>')
        bookmap_parts.append(f'    <topicref href="{preface_filename}"/>')
        bookmap_parts.append('  </frontmatter>')
        log(f"✅ Added preface to bookmap")
    else:
        # Empty frontmatter if no preface
        bookmap_parts.append('  <frontmatter/>')

    # Build hierarchical bookmap structure
    # Parts become <chapter> elements with nested <topicref> for their chapters
    log(f"📝 Building hierarchical bookmap with {len(files_to_process)} topics...")

    current_chapter_open = False

    for href, topic_id, title, topic_filename in files_to_process:
        # Check if this is a Part heading
        if title.startswith("Part "):
            # Close previous chapter if open
            if current_chapter_open:
                bookmap_parts.append('  </chapter>')
                current_chapter_open = False

            # Start new chapter with the Part as its topic
            bookmap_parts.append(f'  <chapter href="{topic_filename}">')
            current_chapter_open = True
            log(f"  📖 Started chapter: {title}")

        # Check if this is README (first non-Part item)
        elif title == "README":
            # README is a standalone chapter
            bookmap_parts.append(f'  <chapter href="{topic_filename}"/>')
            log(f"  📄 Added standalone chapter: {title}")

        # Check if this is an individual appendix (e.g., "Appendix A: ...")
        elif title.startswith("Appendix ") and ":" in title:
            # Close previous chapter if open (end Part V)
            if current_chapter_open:
                bookmap_parts.append('  </chapter>')
                current_chapter_open = False

            # Add appendix as standalone chapter
            bookmap_parts.append(f'  <chapter href="{topic_filename}"/>')
            log(f"  📚 Added appendix chapter: {title}")

        # Everything else is a nested topicref
        else:
            if current_chapter_open:
                bookmap_parts.append(f'    <topicref href="{topic_filename}"/>')
                log(f"    ✓ Added nested topic: {title}")
            else:
                # Fallback: add as standalone chapter if no part is open
                bookmap_parts.append(f'  <chapter href="{topic_filename}"/>')
                log(f"  ⚠️  Added standalone chapter (no open part): {title}")

    # Close final chapter if still open
    if current_chapter_open:
        bookmap_parts.append('  </chapter>')
    bookmap_parts.append('</bookmap>')
    return "\n".join(bookmap_parts), doc_title

# ---------------------------------------------------------------------
def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert Markdown files to DITA topics and generate bookmap',
        epilog='Metadata is loaded from metadata.yaml. Use --title to override the title.'
    )
    parser.add_argument('--title', type=str, help='Title for the DITA bookmap (overrides metadata.yaml)')
    args = parser.parse_args()

    # Clear old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("🚀 Starting generate-dita.py")
    log(f"📂 Reading Markdown files from: {MD_DIR}/")
    log(f"📂 DITA files will be saved to: {DITA_DIR}/")

    # Load metadata
    metadata = load_metadata()

    # Check if MD directory exists
    md_dir_path = Path(MD_DIR)
    if not md_dir_path.exists():
        log(f"❌ Error: {MD_DIR}/ directory not found. Run generate-md.py first.")
        sys.exit(1)

    # Check if toc.md exists
    toc_file = md_dir_path / TOC_MD
    if not toc_file.exists():
        log(f"❌ Error: {toc_file} not found. Run generate-md.py first.")
        sys.exit(1)

    # Parse TOC and extract title
    toc_soup, extracted_title = parse_markdown_toc(str(toc_file))

    # Determine final title (CLI arg > metadata > extracted from TOC > first link > default)
    final_title = args.title or extracted_title
    if args.title:
        log(f"📖 Using title from command line: {args.title}")
    elif extracted_title:
        log(f"📖 Using title extracted from TOC")

    # Create DITA directory
    dita_dir = Path(DITA_DIR)
    dita_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old DITA files (but keep images)
    for dita_file in dita_dir.glob("*.dita"):
        dita_file.unlink()
    for ditamap_file in dita_dir.glob("*.ditamap"):
        ditamap_file.unlink()

    # Build DITA bookmap and generate topics
    bookmap_content, doc_title = build_dita_map(toc_soup, dita_dir, base_path=str(md_dir_path), title=final_title, metadata=metadata)

    # Write DITA bookmap
    bookmap_path = dita_dir / "userguide.ditamap"
    with open(bookmap_path, "w", encoding="utf-8") as f:
        f.write(bookmap_content)
    log(f"✅ Generated DITA bookmap: {bookmap_path}")

    report_long_code_lines(md_dir_path)

    log(f"🎉 Done! Generated DITA files in {DITA_DIR}/")

if __name__ == "__main__":
    main()
