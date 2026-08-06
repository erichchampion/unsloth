#!/usr/bin/env python3
"""Package DITA-OT HTML5 output as an EPUB 3 for Kindle Direct Publishing.

This replaces the Calibre `ebook-convert` step the pipeline used to run.
Calibre rebuilds the stylesheet from scratch: it recomputes every rule, renames
every class to `.calibreN`, and drops classes it finds no rules for. Since the
merged chapter files link `commonltr.css` and `common-extended.css` by relative
path but those files were never copied alongside them, Calibre resolved no
rules for Prism's `.token` classes and stripped them — the shipped EPUBs
carried syntax-highlighting markup with no syntax highlighting and not one
colour declaration. It also emitted EPUB 2 with no navigation document, and
overrode the requested code font-size through its font-rescaling filter.

Owning the packaging outright fixes all of that, and lets the markup be shaped
for what Kindle actually supports rather than for what a browser does.

The Amazon Kindle Publishing Guidelines drive three decisions here:

  * `white-space` is supported only as `nowrap` or `normal` (Appendix B) —
    neither `pre` nor `pre-wrap`. Whitespace inside <pre> cannot be relied on,
    so `transform_codeblocks()` rewrites each code sample into one block-level
    element per logical line with its leading indentation hard-spaced.

  * The book contains both a `nav` document and an NCX. §5.2 asks plainly to
    "make sure your book contains NCX", and §5.3.1 asks for landmarks *and*
    guide items rather than one in place of the other.

  * Fonts must be OTF or TTF (§11.3.8); WOFF is not listed. The embedded faces
    are TTF, subsetted here to the glyphs the book actually uses.
"""

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape
from xml.sax.saxutils import quoteattr

from bs4 import BeautifulSoup, Doctype

# Paths inside the EPUB container.
OEBPS = "OEBPS"
CSS_HREF = "css/ebook.css"
FONT_DIR = "fonts"
IMAGE_DIR = "images"

# Every code sample is rendered in this font, so the subset must cover at least
# printable ASCII even when the current text happens not to use all of it.
ALWAYS_KEEP = set(range(0x20, 0x7F)) | {0x00A0}

MEDIA_TYPES = {
    ".html": "application/xhtml+xml",
    ".xhtml": "application/xhtml+xml",
    ".css": "text/css",
    ".ttf": "application/vnd.ms-opentype",
    ".otf": "application/vnd.ms-opentype",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ncx": "application/x-dtbncx+xml",
}


def _log(message):
    print(message)


def set_logger(fn):
    """Route this module's output through the caller's logger."""
    global _log
    _log = fn


# --------------------------------------------------------------------------
# Code block transformation
# --------------------------------------------------------------------------

def transform_codeblocks(soup):
    """Rewrite code samples so they survive Kindle's whitespace handling.

    DITA-OT emits `<pre class="pre codeblock language-x"><code>` with real
    newlines and real leading spaces, which is correct HTML and correct for
    every conforming EPUB reader. Kindle is the exception: `white-space: pre`
    is not a supported value there, so both the line breaks and the indentation
    are at risk of collapsing — and for Python, losing indentation loses the
    program.

    Each logical line becomes `<span class="cl">`, styled `display: block` (one
    of the values Appendix B does list). Leading spaces become U+00A0. Both
    changes are inert on a conforming reader — inside a real <pre> a
    no-break space renders as a space and a block box breaks the line anyway —
    so nothing is lost off-Kindle.

    Per-line blocks are also what makes the hanging indent in ebook.css work:
    `text-indent` applies to the first formatted line of a block, so a hanging
    indent on the <pre> as a whole would indent every hard line break rather
    than every soft wrap. One block per line inverts that correctly.

    Returns the number of code blocks transformed.
    """
    count = 0

    for pre in soup.find_all("pre"):
        classes = pre.get("class") or []
        if "codeblock" not in classes:
            continue

        # Normalise the class list: keep `codeblock` as the styling hook, the
        # `diagram` marker that decides whether this is sized to fit rather
        # than wrapped, and the language for anyone reading the XHTML. Drop
        # DITA's generic `pre` class, which collides with the element name.
        kept = ["codeblock"] + [
            c for c in classes if c.startswith("language-") or c == "diagram"
        ]
        pre["class"] = kept

        code = pre.find("code")
        container = code if code is not None else pre

        lines = _split_lines(container, soup)

        # DITA-OT ends every code block with a trailing newline; that produces
        # one empty trailing line which would render as a blank line.
        while lines and not lines[-1]:
            lines.pop()

        container.clear()

        for parts in lines:
            span = soup.new_tag("span")
            span["class"] = ["cl"]

            if parts:
                _harden_leading_indent(parts)
            else:
                # Preserve deliberate blank lines between logical stanzas.
                parts = [" "]

            for part in parts:
                span.append(part)
            container.append(span)

        count += 1

    return count


def _split_lines(node, soup):
    """Split a subtree into one list of nodes per source line.

    A newline is not always a bare text node sitting between Prism spans:
    triple-quoted strings and block comments are single tokens spanning several
    lines, so the split has to reach inside the markup. A tag straddling a
    newline is cloned once per line with its classes intact, which keeps the
    highlighting correct on both sides of the break.
    """
    lines = [[]]

    for child in list(node.children):
        if isinstance(child, str):
            parts = str(child).split("\n")
            for index, part in enumerate(parts):
                if index:
                    lines.append([])
                if part:
                    lines[-1].append(part)
        else:
            for index, sub_line in enumerate(_split_lines(child, soup)):
                if index:
                    lines.append([])
                if not sub_line:
                    continue
                clone = soup.new_tag(child.name)
                for key, value in child.attrs.items():
                    clone[key] = value
                for piece in sub_line:
                    clone.append(piece)
                lines[-1].append(clone)

    return lines


def _harden_leading_indent(parts):
    """Harden the indentation at the start of one assembled line, in place.

    The first piece of a line is usually a bare string, but not always: a line
    inside a triple-quoted string or a block comment begins with the cloned
    token span instead, and that indentation is part of the program too. So the
    first text run is located wherever it sits.
    """
    if isinstance(parts[0], str):
        parts[0] = _harden_indent(parts[0])
        return

    node = parts[0]
    for descendant in node.descendants:
        if isinstance(descendant, str):
            hardened = _harden_indent(str(descendant))
            if hardened != descendant:
                descendant.replace_with(hardened)
            return


def _harden_indent(text):
    """Replace a run of leading spaces with no-break spaces (U+00A0).

    Only the leading run is touched. Interior whitespace is left alone: it is
    not what carries structure, and hard-spacing it would suppress the soft
    wrap opportunities that let a long line break somewhere sensible.
    """
    stripped = text.lstrip(" ")
    indent = len(text) - len(stripped)
    if not indent:
        return text
    return " " * indent + stripped


# --------------------------------------------------------------------------
# Font subsetting
# --------------------------------------------------------------------------

def collect_font_codepoints(soups):
    """Every codepoint rendered in the embedded monospace face."""
    codepoints = set(ALWAYS_KEEP)
    for soup in soups:
        for element in soup.find_all(["pre", "code"]):
            codepoints.update(ord(c) for c in element.get_text())
    return codepoints


def subset_font(src, dest, codepoints):
    """Cut a font down to the glyphs the book uses.

    Returns (bytes_before, bytes_after), or None if fontTools is unavailable —
    in which case the caller falls back to embedding the full face, which is
    correct but larger.
    """
    try:
        from fontTools import subset as ft_subset
    except ImportError:
        return None

    options = ft_subset.Options()
    options.layout_features = ["*"]
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.recalc_bounds = True

    font = ft_subset.load_font(str(src), options)
    subsetter = ft_subset.Subsetter(options=options)
    subsetter.populate(unicodes=codepoints)
    subsetter.subset(font)
    ft_subset.save_font(font, str(dest), options)
    font.close()

    return src.stat().st_size, dest.stat().st_size


def report_missing_glyphs(font_path, codepoints):
    """Codepoints the face cannot render.

    A missing glyph in an embedded font is worse than no embedded font at all:
    the reader draws a notdef box rather than falling back for that one
    character, and in a monospace code sample that is a silent corruption.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return set()

    font = TTFont(str(font_path))
    covered = set()
    for table in font["cmap"].tables:
        covered |= set(table.cmap.keys())
    font.close()

    # Combining marks and control characters are not expected to be covered
    # individually and are not worth reporting.
    return {cp for cp in codepoints if cp not in covered and cp >= 0x20}


# --------------------------------------------------------------------------
# XHTML preparation
# --------------------------------------------------------------------------

XHTML_HEAD = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<!DOCTYPE html>\n'
)


def prepare_chapter(path, title):
    """Read a merged chapter file and return (xhtml_text, soup, sections).

    `sections` is the list of (id, title) pairs for the level-2 navigation.
    DITA-OT anchors sections on an empty <span> that precedes the heading, so
    the heading itself gets an id here — a nav entry has to point at something
    the reader can scroll to.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    # The merged chapter files carry their own doctype. XHTML_HEAD supplies
    # one, and two of them is a parse error at the document element.
    for node in list(soup.contents):
        if isinstance(node, Doctype):
            node.extract()

    transform_codeblocks(soup)

    # Point at the stylesheet this writer owns, and drop the dangling links to
    # DITA-OT's generated CSS, which was never copied next to these files.
    for link in soup.find_all("link"):
        link.decompose()
    for style in soup.find_all("style"):
        style.decompose()

    head = soup.head
    if head is None:
        head = soup.new_tag("head")
        soup.html.insert(0, head)

    if not head.find("meta", charset=True):
        meta = soup.new_tag("meta")
        meta["charset"] = "utf-8"
        head.insert(0, meta)

    if head.find("title") is None:
        title_tag = soup.new_tag("title")
        title_tag.string = title
        head.append(title_tag)

    link = soup.new_tag("link")
    link["rel"] = "stylesheet"
    link["type"] = "text/css"
    link["href"] = CSS_HREF
    head.append(link)

    # Section headings for the second navigation level.
    sections = []
    for index, heading in enumerate(soup.find_all("h2"), start=1):
        classes = heading.get("class") or []
        if "sectiontitle" not in classes:
            continue
        if not heading.get("id"):
            heading["id"] = f"{path.stem}_sec{index}"
        text = heading.get_text(" ", strip=True)
        if text:
            sections.append((heading["id"], text))

    html = soup.find("html")
    if html is not None:
        html["xmlns"] = "http://www.w3.org/1999/xhtml"
        html["xml:lang"] = "en"
        html["lang"] = "en"

    return XHTML_HEAD + str(soup), soup, sections


def check_well_formed(text, label):
    """EPUB 3 content documents must be well-formed XML, not merely valid HTML."""
    from xml.etree import ElementTree

    try:
        ElementTree.fromstring(text)
        return True
    except ElementTree.ParseError as exc:
        _log(f"   ⚠️  {label} is not well-formed XML: {exc}")
        return False


# --------------------------------------------------------------------------
# Package documents
# --------------------------------------------------------------------------

def _meta_title(metadata):
    title = metadata.get("title", "Untitled")
    if metadata.get("subtitle"):
        return f"{title}: {metadata['subtitle']}"
    return title


def _identifier(metadata):
    isbn = metadata.get("isbn")
    if isbn:
        return f"urn:isbn:{str(isbn).replace('-', '')}"
    import uuid
    return f"urn:uuid:{uuid.uuid4()}"


def build_opf(metadata, manifest, spine, identifier, cover_id):
    """The package document, EPUB 3.0.

    A legacy <guide> is emitted alongside the landmarks nav because §5.3.1 asks
    for both. EPUBCheck reports `guide` as deprecated in EPUB 3; that is a
    warning, and Kindle greys out the cover and TOC menu entries without it.
    """
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    authors = metadata.get("author") or []
    if isinstance(authors, str):
        authors = [authors]

    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0"'
        ' unique-identifier="bookid" xml:lang="en">',
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">',
        f'    <dc:identifier id="bookid">{xml_escape(identifier)}</dc:identifier>',
        f'    <dc:title>{xml_escape(_meta_title(metadata))}</dc:title>',
        f'    <dc:language>{xml_escape(metadata.get("language", "en"))}</dc:language>',
    ]

    for index, author in enumerate(authors, start=1):
        parts.append(f'    <dc:creator id="creator{index}">{xml_escape(author)}</dc:creator>')
        parts.append(
            f'    <meta refines="#creator{index}" property="role"'
            f' scheme="marc:relators">aut</meta>'
        )

    if metadata.get("publisher"):
        parts.append(f'    <dc:publisher>{xml_escape(metadata["publisher"])}</dc:publisher>')
    if metadata.get("description"):
        parts.append(f'    <dc:description>{xml_escape(metadata["description"])}</dc:description>')
    if metadata.get("rights"):
        parts.append(f'    <dc:rights>{xml_escape(metadata["rights"])}</dc:rights>')

    parts.append(f'    <meta property="dcterms:modified">{modified}</meta>')
    if cover_id:
        # The EPUB 2 cover convention, which Kindle still reads.
        parts.append(f'    <meta name="cover" content="{cover_id}"/>')
    parts.append('  </metadata>')

    parts.append('  <manifest>')
    for item in manifest:
        props = f' properties="{item["properties"]}"' if item.get("properties") else ""
        parts.append(
            f'    <item id="{item["id"]}" href={quoteattr(item["href"])}'
            f' media-type="{item["media_type"]}"{props}/>'
        )
    parts.append('  </manifest>')

    parts.append('  <spine toc="ncx">')
    for idref in spine:
        parts.append(f'    <itemref idref="{idref}"/>')
    parts.append('  </spine>')

    parts.append('  <guide>')
    parts.append('    <reference type="cover" title="Cover" href="titlepage.html"/>')
    parts.append('    <reference type="toc" title="Table of Contents" href="toc.html"/>')
    parts.append('  </guide>')

    parts.append('</package>')
    return "\n".join(parts) + "\n"


def build_nav(metadata, entries):
    """nav.xhtml — the logical TOC plus the landmarks list.

    Kindle supports two levels of nesting (§5.2), so chapters carry their
    sections and stop there.
    """
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!DOCTYPE html>',
        '<html xmlns="http://www.w3.org/1999/xhtml"'
        ' xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">',
        '<head>',
        '  <meta charset="utf-8"/>',
        '  <title>Table of Contents</title>',
        f'  <link rel="stylesheet" type="text/css" href="{CSS_HREF}"/>',
        '</head>',
        '<body>',
        '  <nav epub:type="toc" id="toc">',
        '    <h1>Table of Contents</h1>',
        '    <ol>',
    ]

    for entry in entries:
        parts.append(f'      <li><a href={quoteattr(entry["href"])}>'
                     f'{xml_escape(entry["title"])}</a>')
        if entry["sections"]:
            parts.append('        <ol>')
            for anchor, text in entry["sections"]:
                href = f'{entry["href"]}#{anchor}'
                parts.append(f'          <li><a href={quoteattr(href)}>'
                             f'{xml_escape(text)}</a></li>')
            parts.append('        </ol>')
        parts.append('      </li>')

    parts += [
        '    </ol>',
        '  </nav>',
        '  <nav epub:type="landmarks" hidden="hidden">',
        '    <h1>Landmarks</h1>',
        '    <ol>',
        '      <li><a epub:type="cover" href="titlepage.html">Cover</a></li>',
        '      <li><a epub:type="toc" href="toc.html">Table of Contents</a></li>',
        '      <li><a epub:type="bodymatter" href="'
        + (entries[0]["href"] if entries else "toc.html")
        + '">Start of Content</a></li>',
        '    </ol>',
        '  </nav>',
        '</body>',
        '</html>',
    ]
    return "\n".join(parts) + "\n"


def build_ncx(metadata, entries, identifier):
    """toc.ncx — required by §5.2 even though EPUB 3 supersedes it."""
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">',
        '  <head>',
        f'    <meta name="dtb:uid" content="{xml_escape(identifier)}"/>',
        '    <meta name="dtb:depth" content="2"/>',
        '    <meta name="dtb:totalPageCount" content="0"/>',
        '    <meta name="dtb:maxPageNumber" content="0"/>',
        '  </head>',
        f'  <docTitle><text>{xml_escape(_meta_title(metadata))}</text></docTitle>',
        '  <navMap>',
    ]

    order = 0
    for index, entry in enumerate(entries, start=1):
        order += 1
        parts.append(f'    <navPoint id="nav{index}" playOrder="{order}">')
        parts.append(f'      <navLabel><text>{xml_escape(entry["title"])}</text></navLabel>')
        parts.append(f'      <content src={quoteattr(entry["href"])}/>')
        for sub, (anchor, text) in enumerate(entry["sections"], start=1):
            order += 1
            href = f'{entry["href"]}#{anchor}'
            parts.append(f'      <navPoint id="nav{index}_{sub}" playOrder="{order}">')
            parts.append(f'        <navLabel><text>{xml_escape(text)}</text></navLabel>')
            parts.append(f'        <content src={quoteattr(href)}/>')
            parts.append('      </navPoint>')
        parts.append('    </navPoint>')

    parts += ['  </navMap>', '</ncx>']
    return "\n".join(parts) + "\n"


def build_titlepage(metadata, cover_href):
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<!DOCTYPE html>',
        '<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">',
        '<head>',
        '  <meta charset="utf-8"/>',
        f'  <title>{xml_escape(_meta_title(metadata))}</title>',
        f'  <link rel="stylesheet" type="text/css" href="{CSS_HREF}"/>',
        '</head>',
        '<body>',
    ]
    if cover_href:
        parts.append(
            f'  <div class="cover"><img src={quoteattr(cover_href)}'
            f' alt={quoteattr(_meta_title(metadata))}/></div>'
        )
    else:
        parts.append('  <div class="titlepage">')
        parts.append(f'    <p class="booktitle">{xml_escape(metadata.get("title", ""))}</p>')
        if metadata.get("subtitle"):
            parts.append(f'    <p class="booksubtitle">{xml_escape(metadata["subtitle"])}</p>')
        parts.append('  </div>')
    parts += ['</body>', '</html>']
    return "\n".join(parts) + "\n"


CONTAINER_XML = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<container version="1.0"'
    ' xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <rootfiles>\n'
    f'    <rootfile full-path="{OEBPS}/content.opf"'
    ' media-type="application/oebps-package+xml"/>\n'
    '  </rootfiles>\n'
    '</container>\n'
)


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

def _zip_epub(stage, output_path):
    """Write the container.

    `mimetype` must come first and be stored uncompressed with no extra field —
    that is what lets a reader identify the file from its first bytes.
    """
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w") as zf:
        zf.write(stage / "mimetype", "mimetype", compress_type=zipfile.ZIP_STORED)

        for path in sorted(stage.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(stage).as_posix()
            if rel == "mimetype":
                continue
            zf.write(path, rel, compress_type=zipfile.ZIP_DEFLATED)


def write_epub(metadata, chapter_files, chapter_structure, toc_file,
               style_dir, output_path, work_dir):
    """Assemble the EPUB 3.

    Args:
        metadata: parsed metadata.yaml
        chapter_files: merged chapter HTML paths, in reading order
        chapter_structure: the bookmap structure, parallel to chapter_files
        toc_file: the reading-order HTML table of contents
        style_dir: book/style, holding ebook.css and fonts/
        output_path: where to write the .epub
        work_dir: scratch directory for the staged container

    Returns True on success.
    """
    style_dir = Path(style_dir)
    css_source = style_dir / "ebook.css"
    if not css_source.exists():
        _log(f"❌ Stylesheet not found: {css_source}")
        return False

    stage = Path(work_dir) / "epub-stage"
    if stage.exists():
        shutil.rmtree(stage)
    oebps = stage / OEBPS
    oebps.mkdir(parents=True)
    (stage / "META-INF").mkdir()

    (stage / "mimetype").write_text("application/epub+zip", encoding="utf-8")
    (stage / "META-INF" / "container.xml").write_text(CONTAINER_XML, encoding="utf-8")

    titles = {info["filename"]: info["title"] for info in chapter_structure}

    # ---- chapters -------------------------------------------------------
    entries = []
    soups = []
    well_formed = True

    for path in chapter_files:
        title = titles.get(path.name, path.stem.replace("_", " ").title())
        text, soup, sections = prepare_chapter(path, title)
        soups.append(soup)

        if not check_well_formed(text, path.name):
            well_formed = False

        (oebps / path.name).write_text(text, encoding="utf-8")
        entries.append({"href": path.name, "title": title, "sections": sections})

    if not well_formed:
        _log("   ⚠️  Continuing despite malformed chapters; EPUBCheck will list them")

    # ---- reading-order TOC page ----------------------------------------
    toc_name = "toc.html"
    if toc_file and Path(toc_file).exists():
        toc_text, toc_soup, _ = prepare_chapter(Path(toc_file), "Table of Contents")
        check_well_formed(toc_text, toc_name)
        (oebps / toc_name).write_text(toc_text, encoding="utf-8")
        soups.append(toc_soup)

    # ---- stylesheet -----------------------------------------------------
    (oebps / "css").mkdir()
    shutil.copy2(css_source, oebps / CSS_HREF)

    # ---- fonts ----------------------------------------------------------
    codepoints = collect_font_codepoints(soups)
    font_items = []
    fonts_src = style_dir / "fonts"
    if fonts_src.is_dir():
        (oebps / FONT_DIR).mkdir()
        for font_path in sorted(fonts_src.glob("*.ttf")):
            missing = report_missing_glyphs(font_path, codepoints)
            if missing:
                chars = " ".join(f"U+{cp:04X}({chr(cp)})" for cp in sorted(missing)[:20])
                _log(f"   ⚠️  {font_path.name} cannot render {len(missing)} "
                     f"codepoint(s) used in code: {chars}")

            dest = oebps / FONT_DIR / font_path.name
            result = subset_font(font_path, dest, codepoints)
            if result is None:
                shutil.copy2(font_path, dest)
                _log(f"   ⚠️  fontTools unavailable; embedding {font_path.name} unsubsetted")
            else:
                before, after = result
                _log(f"   ✓ {font_path.name}: {before // 1024} KB → {after // 1024} KB "
                     f"({len(codepoints)} codepoints)")
            font_items.append(f"{FONT_DIR}/{font_path.name}")

    # ---- cover ----------------------------------------------------------
    cover_href = None
    cover_id = None
    cover_path = metadata.get("cover-image")
    if cover_path:
        cover_path = Path(cover_path)
        if cover_path.exists():
            (oebps / IMAGE_DIR).mkdir(exist_ok=True)
            dest = oebps / IMAGE_DIR / cover_path.name
            shutil.copy2(cover_path, dest)
            cover_href = f"{IMAGE_DIR}/{cover_path.name}"
            cover_id = "cover-image"
            _log(f"   ✓ Cover: {cover_path.name}")
        else:
            _log(f"   ⚠️  Cover image not found: {cover_path}")

    (oebps / "titlepage.html").write_text(
        build_titlepage(metadata, cover_href), encoding="utf-8")

    # ---- navigation -----------------------------------------------------
    identifier = _identifier(metadata)
    (oebps / "nav.xhtml").write_text(build_nav(metadata, entries), encoding="utf-8")
    (oebps / "toc.ncx").write_text(build_ncx(metadata, entries, identifier), encoding="utf-8")

    # ---- manifest and spine ---------------------------------------------
    manifest = [
        {"id": "nav", "href": "nav.xhtml",
         "media_type": "application/xhtml+xml", "properties": "nav"},
        {"id": "ncx", "href": "toc.ncx", "media_type": MEDIA_TYPES[".ncx"]},
        {"id": "css", "href": CSS_HREF, "media_type": MEDIA_TYPES[".css"]},
        {"id": "titlepage", "href": "titlepage.html",
         "media_type": MEDIA_TYPES[".html"]},
    ]
    spine = ["titlepage"]

    if cover_href:
        manifest.append({
            "id": cover_id,
            "href": cover_href,
            "media_type": MEDIA_TYPES.get(Path(cover_href).suffix.lower(), "image/jpeg"),
            "properties": "cover-image",
        })

    for index, href in enumerate(font_items, start=1):
        manifest.append({"id": f"font{index}", "href": href,
                         "media_type": MEDIA_TYPES[Path(href).suffix.lower()]})

    if (oebps / toc_name).exists():
        manifest.append({"id": "tocpage", "href": toc_name,
                         "media_type": MEDIA_TYPES[".html"]})
        spine.append("tocpage")

    for index, entry in enumerate(entries, start=1):
        item_id = f"ch{index}"
        manifest.append({"id": item_id, "href": entry["href"],
                         "media_type": MEDIA_TYPES[".html"]})
        spine.append(item_id)

    (oebps / "content.opf").write_text(
        build_opf(metadata, manifest, spine, identifier, cover_id), encoding="utf-8")

    # ---- zip ------------------------------------------------------------
    _zip_epub(stage, output_path)
    size_kb = Path(output_path).stat().st_size / 1024
    _log(f"✅ EPUB 3 written: {Path(output_path).name} ({size_kb:.1f} KB, "
         f"{len(entries)} chapters)")
    return True
