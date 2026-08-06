#!/usr/bin/env python3

import os
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

def combine_with_cover_page(output_pdf: str):
    """
    Post-processing step to combine 8.5x11.pdf with the generated user guide PDF.
    Uses ghostscript to redistill and combine PDFs with 8.5x11.pdf as the first page.
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
        str(cover_pdf),
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
