#!/usr/bin/env python3

import os
import re
import sys
import shutil
import subprocess
import yaml
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
DITA_DIR = "dita"
METADATA_FILE = "metadata.yaml"
# publish-book.zsh runs non-interactively, so the toolkit location cannot
# depend on an interactive shell's PATH.
DITA_COMMAND = os.environ.get("DITA_COMMAND", "dita")
LOG_FILE = "generate-pdf.log"

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
            metadata['title'] = "Building AI Coding Assistants"

        log(f"✅ Loaded metadata from {metadata_file}")
        log(f"   Title: {metadata.get('title')}")
        if metadata.get('author'):
            log(f"   Author: {metadata.get('author')}")

        return metadata
    except FileNotFoundError:
        log(f"⚠️  Warning: {metadata_file} not found, using defaults")
        return {'title': 'Building AI Coding Assistants', 'language': 'en'}
    except yaml.YAMLError as e:
        log(f"⚠️  Warning: Error parsing {metadata_file}: {e}")
        return {'title': 'Building AI Coding Assistants', 'language': 'en'}

def run_dita_ot(ditamap_path: Path, output_dir: Path, output_pdf: str, dita_dir: Path, metadata: dict = None):
    """Run DITA-OT to generate PDF with metadata."""
    log(f"🔨 Running DITA-OT to generate PDF...")

    # Check if DITA-OT command is available
    try:
        result = subprocess.run(
            [DITA_COMMAND, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        log(f"📦 DITA-OT version: {result.stdout.strip()}")
    except FileNotFoundError:
        log(f"❌ Error: DITA-OT command '{DITA_COMMAND}' not found.")
        log("   Please install DITA-OT and ensure 'dita' is in your PATH.")
        log("   Download: https://www.dita-ot.org/download")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        log(f"❌ Error checking DITA-OT version: {e}")
        sys.exit(1)

    # Check if pdf-theme plugin is installed
    try:
        result = subprocess.run(
            [DITA_COMMAND, "plugins"],
            capture_output=True,
            text=True,
            check=True
        )
        if "pdf-theme" not in result.stdout:
            log(f"❌ Error: pdf-theme plugin is not installed.")
            log("   Please run ./install-pdf-theme.sh to install the plugin.")
            sys.exit(1)
        log(f"✓ pdf-theme plugin is installed")
    except subprocess.CalledProcessError as e:
        log(f"❌ Error checking installed plugins: {e}")
        sys.exit(1)

    # There used to be a create_pdf_customization() call here. It wrote a
    # `com.custom.pdf` plugin into dita/pdf-custom/ on every run — 8pt
    # grey-boxed codeblock attribute sets and a TOC override — but the plugin
    # was never installed and never passed to `dita`, so none of it reached the
    # PDF. It also declared the extension point `dita.conductor.xslt.param`,
    # which is not where attribute-set overrides go (`dita.xsl.xslfo` is), so
    # installing it would not have worked either. PDF code styling comes from
    # pdf-theme/cfg/fo/attrs/pr-domain-attr.xsl.

    # Run DITA-OT
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        DITA_COMMAND,
        "-i", str(ditamap_path),
        "-f", "pdf-theme",
        "-o", str(output_dir),
        f"-Dargs.rellinks=none",
        f"-Dpdf.formatter=fop"
    ]

    log(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        log(result.stdout)

        # Find the generated PDF
        pdf_files = list(output_dir.glob("*.pdf"))
        if pdf_files:
            generated_pdf = pdf_files[0]
            # Move to desired output location
            shutil.move(str(generated_pdf), output_pdf)
            log(f"✅ PDF generated: {output_pdf}")
        else:
            log(f"⚠️  PDF generated but not found in {output_dir}")

    except subprocess.CalledProcessError as e:
        log(f"❌ Error running DITA-OT:")
        log(e.stdout)
        log(e.stderr)
        sys.exit(1)

def page_size(gs_command: str, pdf: Path):
    """The first page's MediaBox as (width, height) in points, or None.

    Ghostscript is already a hard requirement here, so the size is read with
    it rather than by adding a second PDF library to the toolchain.
    """
    # The path is interpolated into a PostScript string literal.
    escaped = re.sub(r"([()\\])", r"\\\1", str(pdf))
    query = (
        f"({escaped}) (r) file runpdfbegin 1 pdfgetpage "
        "/MediaBox pget {==} {(none) =} ifelse quit"
    )

    try:
        result = subprocess.run(
            [gs_command, "-q", "-dNODISPLAY", "-dNOSAFER", "-c", query],
            capture_output=True, text=True, check=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    found = re.search(r"\[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*\]",
                      result.stdout)
    if not found:
        return None

    x0, y0, x1, y1 = (float(n) for n in found.groups())
    return abs(x1 - x0), abs(y1 - y0)


def normalise_cover(gs_command: str, cover_pdf: Path, target, work_dir: Path):
    """Rescale the cover to the body's page size, or return it unchanged.

    The cover art is exported at 300 dpi, and its page box carries those
    pixels as if they were points: 2550x3300 against the body's 612x792, the
    same 8.5x11 inches at 4.167 times the size. Acrobat and Preview both
    believe it, so the book opens on a cover page nearly three feet wide.

    The aspect ratios match, so fitting the page is a pure scale — no crop, no
    letterboxing. Comparing renders of the two at the same pixel size, 42 of
    935,000 pixels differ, all of them antialiasing along glyph edges.
    """
    size = page_size(gs_command, cover_pdf)
    if size is None:
        log("   ⚠️  Could not read the cover's page size; using it unchanged")
        return cover_pdf

    # A point either way is well inside what rounding explains.
    if max(abs(size[0] - target[0]), abs(size[1] - target[1])) < 1.0:
        return cover_pdf

    log(f"   Cover is {size[0]:g}x{size[1]:g}pt against the body's "
        f"{target[0]:g}x{target[1]:g}pt; rescaling")

    scaled = work_dir / "cover_letter.pdf"
    cmd = [
        gs_command, "-dBATCH", "-dNOPAUSE", "-q",
        "-sDEVICE=pdfwrite",
        "-dPDFSETTINGS=/prepress",
        f"-dDEVICEWIDTHPOINTS={target[0]:g}",
        f"-dDEVICEHEIGHTPOINTS={target[1]:g}",
        "-dFIXEDMEDIA",   # the page box becomes the size asked for
        "-dPDFFitPage",   # and the art is scaled into it
        f"-sOutputFile={scaled}",
        str(cover_pdf),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        log("   ⚠️  Could not rescale the cover; using it unchanged")
        log(exc.stderr)
        return cover_pdf

    return scaled


def combine_with_cover_page(output_pdf: str):
    """
    Post-processing step to combine 8.5x11.pdf with the generated user guide PDF.
    Uses ghostscript to redistill and combine PDFs with 8.5x11.pdf as the first page.

    The cover is rescaled to the body's page size first — see normalise_cover().
    """
    output_path = Path(output_pdf)
    output_dir = output_path.parent if output_path.parent != Path('.') else Path.cwd()
    cover_pdf = output_dir / "8.5x11.pdf"

    # Check if cover page exists
    if not cover_pdf.exists():
        log(f"ℹ️  No 8.5x11.pdf found in {output_dir}, skipping cover page combination")
        return

    log(f"📄 Found {cover_pdf}, combining with {output_pdf}")

    # Check if ghostscript is available
    gs_command = None
    for cmd in ["gs", "gswin64c", "gswin32c"]:  # Try different gs command names
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            gs_command = cmd
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    if not gs_command:
        log(f"⚠️  Ghostscript (gs) not found. Cannot combine PDFs.")
        log("   Install ghostscript: brew install ghostscript")
        return

    # Match the cover to the body rather than the other way round: the body is
    # whatever trim size the PDF theme was built for, and is already right.
    target = page_size(gs_command, output_path)
    if target is None:
        log("   ⚠️  Could not read the body's page size; leaving the cover alone")
        combined_cover = cover_pdf
    else:
        combined_cover = normalise_cover(
            gs_command, cover_pdf, target, output_path.parent
        )

    # Create temporary output file
    temp_output = output_path.parent / f"{output_path.stem}_combined.pdf"

    # Use ghostscript to combine and redistill PDFs
    cmd = [
        gs_command,
        "-dBATCH",
        "-dNOPAUSE",
        "-q",
        "-sDEVICE=pdfwrite",
        "-dPDFSETTINGS=/prepress",  # High quality output
        f"-sOutputFile={temp_output}",
        str(combined_cover),
        str(output_path)
    ]

    log(f"   Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Replace original PDF with combined version
        shutil.move(str(temp_output), str(output_path))
        log(f"✅ Successfully combined {cover_pdf.name} with {output_pdf}")

    except subprocess.CalledProcessError as e:
        log(f"❌ Error combining PDFs with ghostscript:")
        log(e.stderr)
        # Clean up temp file if it exists
        if temp_output.exists():
            temp_output.unlink()
        log(f"⚠️  Continuing with original PDF (not combined)")

    finally:
        # The rescaled cover is scratch; the original 8.5x11.pdf is not.
        if combined_cover != cover_pdf and combined_cover.exists():
            combined_cover.unlink()

# ---------------------------------------------------------------------
def main():
    # Clear old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("🚀 Starting generate-pdf.py")
    log(f"📂 Reading DITA files from: {DITA_DIR}/")

    # Load metadata
    metadata = load_metadata()

    # Determine output PDF filename from metadata
    output_pdf = f"{metadata.get('title', 'Building AI Coding Assistants')}.pdf"
    log(f"📄 Output PDF: {output_pdf}")

    # Check if DITA directory exists
    dita_dir = Path(DITA_DIR)
    if not dita_dir.exists():
        log(f"❌ Error: {DITA_DIR}/ directory not found. Run generate-dita.py first.")
        sys.exit(1)

    # Check if ditamap exists
    ditamap_path = dita_dir / "userguide.ditamap"
    if not ditamap_path.exists():
        log(f"❌ Error: {ditamap_path} not found. Run generate-dita.py first.")
        sys.exit(1)

    # Clean up old output directory
    out_dir = dita_dir / "out"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Run DITA-OT to generate PDF
    run_dita_ot(ditamap_path, out_dir, output_pdf, dita_dir, metadata)

    # Post-processing: Combine with cover page if it exists
    combine_with_cover_page(output_pdf)

    # Clean up temporary output directory (optional)
    # Uncomment the next lines if you want to remove the temp directory
    # if out_dir.exists():
    #     shutil.rmtree(out_dir)
    #     log(f"🧹 Cleaned up temporary directory: {out_dir}")

    log(f"🎉 Done! PDF saved to {output_pdf}")

if __name__ == "__main__":
    main()
