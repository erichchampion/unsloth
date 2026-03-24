#!/usr/bin/env python3

import os
import sys
import shutil
import subprocess
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DITA_DIR = "dita"
HTML_OUTPUT_DIR = "dita/out-html5"
METADATA_FILE = "metadata.yaml"  # Metadata file shared with PDF generation
DITA_COMMAND = "dita"
EBOOK_CONVERT_COMMAND = "/Applications/calibre.app/Contents/MacOS/ebook-convert"
LOG_FILE = "generate-epub.log"
PRISM_THEME = "solarized"  # Options: default, solarized, bootstrap

# ---------------------------------------------------------------------
def log(message: str):
    """Write message to stdout and append to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write(line + "\n")

def check_command(command: str, install_instructions: str = None) -> bool:
    """Check if a command is available."""
    try:
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_output = result.stdout.split('\n')[0]
        log(f"✓ {command} found: {version_output}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        log(f"❌ Error: '{command}' command not found")
        if install_instructions:
            log(f"   {install_instructions}")
        return False

def generate_html5(ditamap_path: Path, output_dir: Path) -> bool:
    """Generate HTML5 output with Prism.js syntax highlighting."""
    log("🔨 Generating HTML5 with syntax highlighting...")

    # Check if Prism.js plugin is installed
    try:
        result = subprocess.run(
            [DITA_COMMAND, "plugins"],
            capture_output=True,
            text=True,
            check=True
        )
        if "fox.jason.prismjs" in result.stdout:
            log(f"✓ Prism.js syntax highlighting plugin is installed")
            has_prismjs = True
        else:
            log(f"⚠️  Warning: Prism.js plugin not installed - code blocks will not have syntax highlighting")
            log("   Install with: dita install https://github.com/jason-fox/fox.jason.prismjs/archive/master.zip")
            has_prismjs = False
    except subprocess.CalledProcessError as e:
        log(f"❌ Error checking installed plugins: {e}")
        return False

    # Clean old output
    if output_dir.exists():
        shutil.rmtree(output_dir)
        log(f"🧹 Cleaned old HTML output: {output_dir}")

    # Build DITA-OT command
    cmd = [
        DITA_COMMAND,
        "--input", str(ditamap_path),
        "--format", "html5",
        "--output", str(output_dir)
    ]

    # Add Prism.js theme if plugin is available
    if has_prismjs:
        cmd.append(f"-Dprism.use.theme={PRISM_THEME}")
        log(f"🎨 Using Prism.js theme: {PRISM_THEME}")

    log(f"   Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Check for warnings
        if "Warning" in result.stdout:
            for line in result.stdout.split('\n'):
                if "Warning" in line:
                    log(line)

        log(f"✅ HTML5 generated successfully")
        return True

    except subprocess.CalledProcessError as e:
        log(f"❌ Error generating HTML5:")
        log(e.stdout)
        log(e.stderr)
        return False

def load_metadata(metadata_file: Path) -> dict:
    """Load EPUB metadata from yaml file."""
    log("📝 Loading EPUB metadata...")

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)

        # Validate required fields
        required_fields = ['title', 'language']
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                log(f"❌ Error: Required field '{field}' missing in {metadata_file}")
                return None

        log(f"✅ Metadata loaded: {metadata['title']}")
        if metadata.get('author'):
            log(f"   Author: {metadata['author']}")

        return metadata
    except FileNotFoundError:
        log(f"❌ Error: Metadata file not found: {metadata_file}")
        log(f"   Please ensure {METADATA_FILE} exists in the book directory")
        return None
    except yaml.YAMLError as e:
        log(f"❌ Error parsing metadata YAML: {e}")
        return None
    except Exception as e:
        log(f"❌ Error loading metadata: {e}")
        return None

def collect_html_files(html_dir: Path) -> list:
    """Collect HTML files in order."""
    log("📚 Collecting HTML files...")

    # Collect all topic files (topic_1.html, topic_2.html, etc.)
    # These contain the actual content with syntax highlighting
    topic_files = sorted(
        html_dir.glob("topic_*.html"),
        key=lambda x: int(x.stem.split('_')[1]) if '_' in x.stem and x.stem.split('_')[1].isdigit() else 0
    )

    if topic_files:
        log(f"✓ Found {len(topic_files)} topic files with content")
        # Optional: include index.html as the first "chapter" if it has content
        index_file = html_dir / "index.html"
        if index_file.exists():
            # Check if index has meaningful content (more than just navigation)
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '<p>' in content or '<h1>' in content:  # Has actual content
                        topic_files.insert(0, index_file)
                        log(f"✓ Including index.html as introduction")
            except:
                pass

        return topic_files
    else:
        # Fallback: just use index.html if no topic files
        index_file = html_dir / "index.html"
        if index_file.exists():
            log(f"⚠️  Only found index.html (may not have full content)")
            return [index_file]
        else:
            log(f"❌ No HTML files found in {html_dir}")
            return []

def extract_title_from_dita(dita_path: Path) -> str:
    """Extract the title from a DITA topic file.

    Args:
        dita_path: Path to the DITA file

    Returns:
        The title text from the <title> element, or a default if not found
    """
    try:
        tree = ET.parse(dita_path)
        root = tree.getroot()

        # Remove namespace prefixes for easier parsing
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        # Find the title element (should be at /topic/title or /concept/title, etc.)
        title_elem = root.find('.//title')
        if title_elem is not None and title_elem.text:
            return title_elem.text.strip()

        # Fallback to filename if no title found
        return dita_path.stem

    except Exception as e:
        log(f"   ⚠️  Warning: Could not extract title from {dita_path.name}: {e}")
        return dita_path.stem

def parse_bookmap_structure(ditamap_path: Path) -> list:
    """Parse bookmap XML to extract chapter structure.

    Returns a list of chapter dictionaries with structure:
    [
        {
            'type': 'frontmatter',
            'title': 'Front Matter',
            'topics': ['preface.dita'],
            'filename': 'frontmatter.html'
        },
        {
            'type': 'chapter',
            'number': 1,
            'title': 'Chapter Title',
            'topics': ['topic_1.dita', 'topic_2.dita', ...],
            'filename': 'chapter_1.html'
        },
        ...
    ]
    """
    log("📖 Parsing bookmap structure...")

    # Get the directory containing the ditamap to resolve relative DITA paths
    dita_dir = ditamap_path.parent

    try:
        tree = ET.parse(ditamap_path)
        root = tree.getroot()

        # Remove namespace prefixes for easier parsing
        # DITA uses various namespaces that can complicate XPath queries
        for elem in root.iter():
            # Strip namespace from tag
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        chapters = []
        chapter_number = 0

        # Parse frontmatter if present
        frontmatter = root.find('.//frontmatter')
        if frontmatter is not None:
            topics = []
            for topicref in frontmatter.findall('.//topicref'):
                href = topicref.get('href')
                if href:
                    topics.append(href)

            if topics:
                chapters.append({
                    'type': 'frontmatter',
                    'title': 'Front Matter',
                    'topics': topics,
                    'filename': 'frontmatter.html'
                })
                log(f"   ✓ Front Matter: {len(topics)} topics")

        # Parse chapters
        # First, check if there are direct chapter elements
        chapter_elems = root.findall('.//chapter')

        # Track all topics that have been used to avoid duplicates
        all_used_topics = set()

        if chapter_elems:
            # Process each chapter element - new hierarchical structure
            for chapter_elem in chapter_elems:
                # Get the chapter's own href (e.g., <chapter href="topic_2.dita">)
                chapter_href = chapter_elem.get('href')

                # Get child topicrefs (chapters nested under Parts)
                child_topicrefs = [t for t in chapter_elem if t.tag == 'topicref']

                if chapter_href:
                    # Chapter element has an href - it's either a Part or standalone chapter
                    if child_topicrefs:
                        # This is a Part with nested chapters
                        # Add the Part page itself
                        if chapter_href not in all_used_topics:
                            chapter_number += 1
                            title = extract_title_from_dita(dita_dir / chapter_href)
                            chapters.append({
                                'type': 'chapter',
                                'number': chapter_number,
                                'title': title,
                                'topics': [chapter_href],
                                'filename': f'chapter_{chapter_number}.html'
                            })
                            all_used_topics.add(chapter_href)
                            log(f"   ✓ Chapter {chapter_number}: {title} (Part)")

                        # Then add each nested chapter
                        for child_topicref in child_topicrefs:
                            child_href = child_topicref.get('href')
                            if child_href and child_href not in all_used_topics:
                                chapter_number += 1
                                title = extract_title_from_dita(dita_dir / child_href)
                                chapters.append({
                                    'type': 'chapter',
                                    'number': chapter_number,
                                    'title': title,
                                    'topics': [child_href],
                                    'filename': f'chapter_{chapter_number}.html'
                                })
                                all_used_topics.add(child_href)
                                log(f"   ✓ Chapter {chapter_number}: {title}")
                    else:
                        # Standalone chapter (README or Appendix)
                        if chapter_href not in all_used_topics:
                            chapter_number += 1
                            title = extract_title_from_dita(dita_dir / chapter_href)
                            chapters.append({
                                'type': 'chapter',
                                'number': chapter_number,
                                'title': title,
                                'topics': [chapter_href],
                                'filename': f'chapter_{chapter_number}.html'
                            })
                            all_used_topics.add(chapter_href)
                            log(f"   ✓ Chapter {chapter_number}: {title} (standalone)")
                else:
                    # Chapter element has NO href - old structure with nested topicrefs
                    # This handles backward compatibility
                    for child in chapter_elem:
                        if child.tag == 'topicref':
                            href = child.get('href')
                            if href and href not in all_used_topics:
                                chapter_number += 1
                                title = extract_title_from_dita(dita_dir / href)
                                chapters.append({
                                    'type': 'chapter',
                                    'number': chapter_number,
                                    'title': title,
                                    'topics': [href],
                                    'filename': f'chapter_{chapter_number}.html'
                                })
                                all_used_topics.add(href)
                                log(f"   ✓ Chapter {chapter_number}: {title} (legacy)")

        # Parse backmatter if present
        backmatter = root.find('.//backmatter')
        if backmatter is not None:
            topics = []
            for topicref in backmatter.findall('.//topicref'):
                href = topicref.get('href')
                if href:
                    topics.append(href)

            if topics:
                chapters.append({
                    'type': 'backmatter',
                    'title': 'Back Matter',
                    'topics': topics,
                    'filename': 'backmatter.html'
                })
                log(f"   ✓ Back Matter: {len(topics)} topics")

        log(f"✅ Parsed {len(chapters)} sections from bookmap")
        return chapters

    except ET.ParseError as e:
        log(f"❌ Error parsing bookmap XML: {e}")
        return []
    except Exception as e:
        log(f"❌ Error parsing bookmap: {e}")
        return []

def merge_html_by_chapter(chapter_structure: list, html_dir: Path, output_dir: Path) -> tuple[list, dict]:
    """Merge HTML files by chapter.

    Args:
        chapter_structure: List of chapter dicts from parse_bookmap_structure()
        html_dir: Directory containing topic HTML files
        output_dir: Directory to write merged chapter HTML files

    Returns:
        Tuple of (list of chapter HTML file paths, topic-to-chapter mapping dict)
    """
    log("🔨 Merging HTML files by chapter...")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapping of DITA file to HTML file using file_to_topic.json
    # The files have been renamed from topic_N.html to their original markdown names
    dita_to_html = {}

    # Load the mapping from file_to_topic.json
    mapping_file = Path(DITA_DIR) / "file_to_topic.json"
    if mapping_file.exists():
        import json
        with open(mapping_file, 'r', encoding='utf-8') as f:
            file_mapping = json.load(f)

        # Build dita_to_html mapping: topic_N -> original-name.html
        for dita_file, info in file_mapping.items():
            topic_base = Path(dita_file).stem  # e.g., "topic_1"
            html_name = info['original_name'] + '.html'
            html_file = html_dir / html_name
            if html_file.exists():
                dita_to_html[topic_base] = html_file
                log(f"   📄 Mapped {topic_base} -> {html_name}")
            else:
                log(f"   ⚠️  Warning: Mapped {topic_base} -> {html_name} but file not found")

    # Also map other HTML files by exact name match (like preface_copyright)
    for html_file in html_dir.glob("*.html"):
        stem = html_file.stem
        if stem not in dita_to_html:
            dita_to_html[stem] = html_file
            log(f"   📄 Mapped {html_file.name} by name")

    chapter_files = []
    topic_to_chapter = {}  # Maps topic_N.html to chapter_M.html
    global_topic_id = 1

    for chapter_info in chapter_structure:
        chapter_filename = chapter_info['filename']
        chapter_file = output_dir / chapter_filename

        # Create merged HTML for this chapter
        merged_soup = BeautifulSoup('<!DOCTYPE html><html><head><meta charset="utf-8"/></head><body></body></html>', 'html.parser')
        body = merged_soup.body

        # Add title tag in head for EPUB readers
        title_tag = merged_soup.new_tag('title')
        title_tag.string = chapter_info['title']
        merged_soup.head.append(title_tag)

        # Don't add a chapter heading h1 here - the DITA topic content already includes
        # the h1 heading, so adding another would create a duplicate

        # Process each topic in this chapter
        for topic_href in chapter_info['topics']:
            # Find corresponding HTML file
            # The topic_href is like "topic_1.dita" from the bookmap
            # We need to find the corresponding topic_N.html file
            topic_base = Path(topic_href).stem  # e.g., "topic_1"
            topic_html = dita_to_html.get(topic_base)

            if not topic_html or not topic_html.exists():
                log(f"   ⚠️  Warning: HTML file not found for {topic_href}")
                continue

            # Read the topic HTML
            try:
                with open(topic_html, 'r', encoding='utf-8') as f:
                    topic_soup = BeautifulSoup(f.read(), 'html.parser')

                # Merge CSS from head (collect all stylesheets)
                if topic_soup.head:
                    for link in topic_soup.head.find_all('link', rel='stylesheet'):
                        if not merged_soup.head.find('link', href=link.get('href')):
                            merged_soup.head.append(link)
                    for style in topic_soup.head.find_all('style'):
                        merged_soup.head.append(style)

                # Extract body content and wrap in div (without ID to avoid duplication during Calibre splitting)
                if topic_soup.body:
                    topic_div = merged_soup.new_tag('div')

                    # Copy all content from topic body to the div
                    for child in topic_soup.body.children:
                        if child.name:  # Skip text nodes
                            topic_div.append(child.extract())

                    # Put the unique ID on the first h1 (not the div) so it won't be duplicated when Calibre splits files
                    first_h1 = topic_div.find('h1')
                    if first_h1:
                        # Replace DITA's generic ID with our unique topic ID
                        first_h1['id'] = f'topic_{global_topic_id}'

                        # Add class based on chapter type for Calibre TOC generation
                        title = chapter_info['title']
                        if title.startswith('Part '):
                            toc_class = 'part-title'
                        elif title.startswith('Appendix '):
                            toc_class = 'appendix-title'
                        else:
                            toc_class = 'chapter-title'

                        first_h1['class'] = first_h1.get('class', []) + [toc_class]

                    body.append(topic_div)

                    # Record mapping
                    topic_to_chapter[topic_html.name] = chapter_filename
                    global_topic_id += 1

            except Exception as e:
                log(f"   ⚠️  Warning: Error processing {topic_html.name}: {e}")
                continue

        # Unwrap semantic elements for ebook compatibility
        for tag in ['article', 'main', 'nav', 'section']:
            for element in merged_soup.find_all(tag):
                element.unwrap()

        # Add CSS to preserve TOC classes (prevents Calibre from stripping them)
        # Use a minimal property so Calibre doesn't consider the rules "empty"
        style_tag = merged_soup.new_tag('style')
        style_tag.string = 'h1.part-title, h1.chapter-title, h1.appendix-title { display: block; }'
        merged_soup.head.append(style_tag)

        # Write merged chapter file
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(str(merged_soup))

        chapter_files.append(chapter_file)
        log(f"   ✓ {chapter_filename}: merged {len(chapter_info['topics'])} topics")

    log(f"✅ Generated {len(chapter_files)} chapter HTML files")
    return chapter_files, topic_to_chapter

def fix_fragment_anchors(chapter_files: list) -> int:
    """Fix fragment-only anchor hrefs to include DITA-OT's topic prefix.

    DITA-OT prefixes anchor IDs with 'topic_N__' but doesn't update fragment hrefs.
    This function finds fragment-only links (#anchor) and adds the topic prefix
    based on which topic div the link is inside.

    Args:
        chapter_files: List of chapter HTML file paths

    Returns:
        Number of links fixed
    """
    log("🔗 Fixing fragment-only anchor links...")

    fixed_count = 0

    for chapter_file in chapter_files:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        modified = False

        # Find all links with fragment-only hrefs
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Only process fragment-only anchors (start with #, no file part)
            if href.startswith('#') and '/' not in href:
                # Find the parent topic div (has class containing the topic ID)
                # The divs created by merge_html_by_chapter don't have IDs, but the
                # content inside does. Look for a span with id starting with topic_

                # Find any element with an id that starts with "topic_" going upward
                parent = link.parent
                topic_prefix = None

                while parent and parent.name:
                    # Check this element and its descendants for a topic-prefixed id
                    for elem in parent.find_all(id=True):
                        elem_id = elem.get('id', '')
                        if elem_id.startswith('topic_') and '__' in elem_id:
                            # Extract the topic prefix (e.g., "topic_14__")
                            topic_prefix = elem_id.split('__')[0] + '__'
                            break

                    if topic_prefix:
                        break

                    # Also check if this element itself has a topic ID
                    parent_id = parent.get('id', '')
                    if parent_id.startswith('topic_') and '__' in parent_id:
                        topic_prefix = parent_id.split('__')[0] + '__'
                        break

                    parent = parent.parent

                if topic_prefix:
                    # Update the href to include the topic prefix
                    anchor_id = href[1:]  # Remove #
                    new_href = f"#{topic_prefix}{anchor_id}"
                    link['href'] = new_href
                    modified = True
                    fixed_count += 1

        # Save if modified
        if modified:
            with open(chapter_file, 'w', encoding='utf-8') as f:
                f.write(str(soup))

    log(f"✅ Fixed {fixed_count} fragment-only anchor links")
    return fixed_count

def fix_epub_validation_errors(chapter_files: list) -> int:
    """Fix EPUB validation errors in chapter files.

    Fixes:
    1. Blockquotes with inline elements - wraps content in <p> tags
    2. Empty body tags - adds minimal content

    Args:
        chapter_files: List of chapter HTML file paths

    Returns:
        Number of files fixed
    """
    log("🔧 Fixing EPUB validation errors...")

    fixed_count = 0

    for chapter_file in chapter_files:
        try:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            modified = False

            # Fix 1: Wrap inline content in blockquotes with <p> tags
            for blockquote in soup.find_all('blockquote'):
                # Check if blockquote contains only inline elements (em, strong, etc.) or text
                has_block_element = any(child.name in ['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'pre']
                                       for child in blockquote.children if hasattr(child, 'name'))

                if not has_block_element:
                    # Extract all content
                    contents = list(blockquote.children)
                    # Clear blockquote
                    blockquote.clear()
                    # Create p tag and add content to it
                    p_tag = soup.new_tag('p')
                    for content in contents:
                        p_tag.append(content)
                    # Add p tag to blockquote
                    blockquote.append(p_tag)
                    modified = True

            # Fix 2: Check for empty body tags
            if soup.body and len(list(soup.body.children)) == 0:
                # Add a minimal paragraph
                p_tag = soup.new_tag('p')
                p_tag.string = ' '
                soup.body.append(p_tag)
                modified = True

            # Save if modified
            if modified:
                with open(chapter_file, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
                fixed_count += 1

        except Exception as e:
            log(f"   ⚠️  Warning: Error processing {chapter_file.name}: {e}")
            continue

    log(f"✅ Fixed EPUB validation errors in {fixed_count} files")
    return fixed_count

def rename_html_files_to_original_names(html_dir: Path, dita_dir: Path) -> dict:
    """Rename topic_N.html files to match original markdown filenames.

    Reads the file_to_topic.json mapping created by generate-dita.py and renames
    HTML files accordingly. Also updates all internal href links to use new names.

    Args:
        html_dir: Directory containing topic_N.html files
        dita_dir: Directory containing file_to_topic.json mapping

    Returns:
        Dict mapping old filenames to new filenames
    """
    import json
    import shutil
    from bs4 import BeautifulSoup

    log("🔄 Renaming HTML files to match original markdown names...")

    mapping_file = dita_dir / "file_to_topic.json"
    if not mapping_file.exists():
        log(f"⚠️  Warning: {mapping_file} not found, skipping rename")
        return {}

    # Load mapping
    with open(mapping_file, 'r', encoding='utf-8') as f:
        topic_mapping = json.load(f)

    # Build rename map: old_name → new_name
    rename_map = {}
    for topic_file, info in topic_mapping.items():
        old_name = Path(topic_file).stem + ".html"  # topic_1.html
        new_name = info['original_name'] + ".html"   # README.html
        rename_map[old_name] = new_name

    # Also handle preface
    if (html_dir / "preface_copyright.html").exists():
        rename_map["preface_copyright.html"] = "copyright.html"

    # Step 1: Rename files
    for old_name, new_name in rename_map.items():
        old_path = html_dir / old_name
        new_path = html_dir / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            log(f"   ✓ {old_name} → {new_name}")

    # Step 2: Update all href links in HTML files
    log("🔗 Updating internal links in HTML files...")
    html_files = list(html_dir.glob("*.html"))

    for html_file in html_files:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            modified = False
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Check if this is an internal link to a renamed file
                if not href.startswith(('http://', 'https://', '#', '/')):
                    # Extract filename and anchor
                    if '#' in href:
                        file_part, anchor = href.split('#', 1)
                        if file_part in rename_map:
                            link['href'] = f"{rename_map[file_part]}#{anchor}"
                            modified = True
                    elif href in rename_map:
                        link['href'] = rename_map[href]
                        modified = True

            if modified:
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(str(soup))
        except Exception as e:
            log(f"   ⚠️  Warning: Error updating links in {html_file.name}: {e}")

    log(f"✅ Renamed {len(rename_map)} HTML files to match markdown names")
    return rename_map

def create_toc_html(chapter_structure: list, chapter_files: list, output_dir: Path, metadata: dict, html_dir: Path) -> Path:
    """Create a table of contents HTML file with Part/Chapter hierarchy matching TOC.md.

    This creates a properly structured TOC with Parts as top-level sections and
    chapters nested within them, matching the original TOC.md structure.

    Args:
        chapter_structure: List of chapter dicts from parse_bookmap_structure()
        chapter_files: List of generated chapter HTML file paths
        output_dir: Directory containing chapter HTML files
        metadata: Book metadata
        html_dir: Directory with original topic HTML files (to extract titles)

    Returns:
        Path to the TOC HTML file
    """
    log("📑 Creating hierarchical table of contents matching TOC.md structure...")

    toc_file = output_dir / "toc.html"

    # Build mapping of all HTML filenames to their titles
    topic_titles = {}
    all_html_files = [f for f in html_dir.glob("*.html")
                      if not f.name.startswith('index') and f.name != 'toc.html']

    for topic_file in all_html_files:
        try:
            with open(topic_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            title_elem = soup.find('h1')
            if not title_elem:
                title_elem = soup.find('title')
            if title_elem:
                topic_titles[topic_file.name] = title_elem.get_text().strip()
            else:
                topic_titles[topic_file.name] = topic_file.stem.replace('_', ' ').title()
        except Exception as e:
            log(f"   ⚠️  Warning: Could not extract title from {topic_file.name}: {e}")
            topic_titles[topic_file.name] = topic_file.stem.replace('_', ' ').title()

    # Build Part structure dynamically from chapter_structure
    # Identify Parts and group chapters under them
    part_structure = []
    current_part = None

    for idx, ch in enumerate(chapter_structure):
        title = ch.get('title', '')

        # Check if this is a Part heading
        if title.startswith('Part '):
            # Save previous part if exists
            if current_part is not None:
                part_structure.append(current_part)

            # Start new part
            current_part = {
                'title': title,
                'intro_idx': idx,
                'chapter_indices': []
            }
        # Check if we're currently in a Part and this is a regular chapter
        elif current_part is not None and not title.startswith('Appendix ') and ch['type'] != 'frontmatter' and title != 'README':
            # Add this chapter to the current Part
            current_part['chapter_indices'].append(idx)

    # Don't forget to add the last part
    if current_part is not None:
        part_structure.append(current_part)

    log(f"   Built dynamic structure: {len(part_structure)} Parts")

    # Create TOC HTML with hierarchical structure
    html = ['<!DOCTYPE html>', '<html>', '<head>', '<meta charset="utf-8"/>', '<title>Table of Contents</title>']
    html.append('<style>')
    html.append('body { font-family: serif; margin: 2em; line-height: 1.6; }')
    html.append('h1 { text-align: center; margin-bottom: 1em; }')
    html.append('h2 { margin-top: 1.5em; margin-bottom: 0.5em; }')
    html.append('ul { list-style-type: none; padding-left: 0; }')
    html.append('ul ul { padding-left: 2em; margin-top: 0.5em; }')
    html.append('li { margin: 0.4em 0; }')
    html.append('a { text-decoration: none; color: #0066cc; }')
    html.append('a:hover { text-decoration: underline; }')
    html.append('.toc-part-heading { font-weight: bold; font-size: 1.2em; color: #000; }')
    html.append('.chapter-link { font-size: 1.0em; }')
    html.append('.toc-appendix-heading { font-weight: bold; font-size: 1.1em; margin-top: 1em; }')
    html.append('</style>')
    html.append('</head>')
    html.append('<body>')

    book_title = metadata.get('title', 'Table of Contents')
    html.append(f'<h1>{book_title}</h1>')
    html.append('<h2>Table of Contents</h2>')
    html.append('<ul>')

    # Add frontmatter
    if chapter_structure and chapter_structure[0]['type'] == 'frontmatter':
        ch = chapter_structure[0]
        title = ch.get('title', 'Copyright Information')
        html.append(f'<li><a href="{ch["filename"]}" class="chapter-link">{title}</a></li>')

    # Add README and other standalone chapters before Parts
    for idx, ch in enumerate(chapter_structure):
        title = ch.get('title', '')
        # Add standalone chapters (README, etc.) that come before Parts
        if not title.startswith('Part ') and not title.startswith('Appendix ') and ch['type'] != 'frontmatter':
            # Check if this chapter is not part of any Part
            is_in_part = any(idx in part['chapter_indices'] for part in part_structure)
            if not is_in_part and idx < (part_structure[0]['intro_idx'] if part_structure else 999):
                html.append(f'<li><a href="{ch["filename"]}" class="chapter-link">{title}</a></li>')

    # Add Parts with nested chapters
    for part in part_structure:
        # Get Part intro page
        intro_idx = part['intro_idx']
        if intro_idx < len(chapter_structure):
            part_ch = chapter_structure[intro_idx]
            html.append(f'<li>')
            # Use 'toc-part-heading' class instead of 'part-title' to avoid conflict with actual Part content pages
            html.append(f'<h1 class="toc-part-heading"><a href="{part_ch["filename"]}">{part["title"]}</a></h1>')
            html.append('<ul>')

            # Add chapters within this Part
            for ch_idx in part['chapter_indices']:
                if ch_idx < len(chapter_structure):
                    ch = chapter_structure[ch_idx]
                    title = ch.get('title', f'Chapter {ch_idx}')
                    html.append(f'<li><a href="{ch["filename"]}" class="chapter-link">{title}</a></li>')

            html.append('</ul>')
            html.append('</li>')

    # Add Appendices - find them dynamically
    appendices = []
    for idx, ch in enumerate(chapter_structure):
        title = ch.get('title', '')
        if title.startswith('Appendix '):
            appendices.append((idx, ch))

    if appendices:
        html.append('<li>')
        # Use 'toc-appendix-heading' class to avoid conflict with actual Appendix content pages
        html.append('<h1 class="toc-appendix-heading">Appendices</h1>')
        html.append('<ul>')

        for idx, ch in appendices:
            title = ch.get('title', f'Appendix {idx}')
            html.append(f'<li><a href="{ch["filename"]}" class="chapter-link">{title}</a></li>')

        html.append('</ul>')
        html.append('</li>')

    html.append('</ul>')
    html.append('</body>')
    html.append('</html>')

    # Write TOC file
    with open(toc_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))

    # Count chapters and appendices
    num_chapters = sum(len(part['chapter_indices']) for part in part_structure)
    num_appendices = len(appendices)

    log(f"✅ Created hierarchical TOC with {len(part_structure)} Parts, {num_chapters} Chapters, and {num_appendices} Appendices")
    return toc_file

def convert_cross_references(chapter_files: list, topic_to_chapter: dict, chapter_structure: list = None) -> int:
    """Convert cross-references to use chapter-based URLs.

    Args:
        chapter_files: List of chapter HTML file paths
        topic_to_chapter: Mapping of topic_N.html to chapter_M.html
        chapter_structure: List of chapter info dicts with titles and filenames

    Returns:
        Count of converted links
    """
    log("🔗 Converting cross-references...")

    # Build mapping from title-based filenames to chapter filenames
    # e.g., "chapter-01-introduction.html" -> "chapter_3.html"
    title_to_chapter = {}

    # Hardcoded mapping for known legacy filenames (from DITA cross-references)
    legacy_mappings = {
        'chapter-01-introduction.html': None,  # Will be filled below
        'chapter-02-multi-provider.html': None,
        'chapter-03-dependency-injection.html': None,
        'chapter-04-tool-orchestration.html': None,
        'chapter-05-streaming.html': None,
        'chapter-06-conversation.html': None,
        'chapter-07-vcs-intelligence.html': None,
        'chapter-08-interactive-modes.html': None,
        'chapter-09-security.html': None,
        'chapter-10-testing.html': None,
        'chapter-11-performance.html': None,
        'chapter-12-monitoring.html': None,
        'chapter-13-plugin-architecture.html': None,
        'chapter-14-ide-integration.html': None,
        'chapter-15-building-your-own.html': None,
        'part-1-foundations.html': None,
        'part-2-core-architecture.html': None,
        'part-3-advanced-features.html': None,
        'part-4-production-readiness.html': None,
        'part-5-extensibility.html': None,
    }

    if chapter_structure:
        for ch in chapter_structure:
            title = ch.get('title', '')
            filename = ch.get('filename', '')

            # Map chapter numbers to filenames for legacy mapping
            if title.startswith('Chapter '):
                parts = title.split(':', 1)
                if len(parts) == 2:
                    chapter_num_str = parts[0].replace('Chapter ', '').strip()
                    try:
                        chapter_num = int(chapter_num_str)
                        # Update legacy_mappings with actual filename
                        for legacy_name in list(legacy_mappings.keys()):
                            if legacy_name.startswith(f'chapter-{chapter_num:02d}-'):
                                legacy_mappings[legacy_name] = filename
                                break
                    except ValueError:
                        pass
            elif title.startswith('Part '):
                parts = title.split(':', 1)
                if len(parts) == 2:
                    part_num = parts[0].replace('Part ', '').strip()
                    # Convert Roman numerals to numbers
                    roman_to_num = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
                    if part_num in roman_to_num:
                        # Update legacy_mappings with actual filename
                        for legacy_name in list(legacy_mappings.keys()):
                            if legacy_name.startswith(f'part-{roman_to_num[part_num]}-'):
                                legacy_mappings[legacy_name] = filename
                                break

        # Add legacy mappings to title_to_chapter
        for legacy_name, filename in legacy_mappings.items():
            if filename:
                title_to_chapter[legacy_name] = filename

    # Build reverse mapping: topic_N.html -> topic_ID (from the div IDs)
    topic_anchors = {}  # Maps topic_N.html -> set of anchor IDs in that file
    for chapter_file in chapter_files:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Find all topic divs
        for div in soup.find_all('div', id=True):
            div_id = div.get('id')
            if div_id and div_id.startswith('topic_'):
                # This represents a topic that was merged
                # We need to track which topic files point to which IDs
                # For now, we'll just track the IDs available in each chapter
                pass

    converted_count = 0

    for chapter_file in chapter_files:
        with open(chapter_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        chapter_filename = chapter_file.name
        modified = False

        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Skip external links
            if href.startswith(('http://', 'https://', '//', 'mailto:')):
                continue

            # Convert internal .html links
            if href.endswith('.html'):
                # Strip any fragment identifier
                if '#' in href:
                    target_file, fragment = href.split('#', 1)
                else:
                    target_file = href
                    fragment = None

                # First check if this is a legacy filename (chapter-01-introduction.html, part-1-foundations.html, etc.)
                target_chapter = title_to_chapter.get(target_file)

                if target_chapter:
                    # Found in legacy mapping
                    if target_chapter == chapter_filename:
                        # Same chapter - use fragment only if present
                        link['href'] = f'#{fragment}' if fragment else '#'
                    else:
                        # Different chapter
                        link['href'] = f'{target_chapter}#{fragment}' if fragment else target_chapter

                    modified = True
                    converted_count += 1

                # Otherwise check if this is a topic_N.html link
                elif target_file in topic_to_chapter:
                    target_chapter = topic_to_chapter[target_file]

                    # Extract topic number from filename (topic_5.html -> 5)
                    try:
                        topic_num = int(target_file.replace('topic_', '').replace('.html', ''))
                        anchor_id = f'topic_{topic_num}'

                        if target_chapter == chapter_filename:
                            # Same chapter - use anchor only
                            link['href'] = f'#{anchor_id}'
                        else:
                            # Different chapter - use file + anchor
                            link['href'] = f'{target_chapter}#{anchor_id}'

                        modified = True
                        converted_count += 1
                    except ValueError:
                        log(f"   ⚠️  Warning: Could not parse topic number from {target_file}")

        # Save if modified
        if modified:
            with open(chapter_file, 'w', encoding='utf-8') as f:
                f.write(str(soup))

    log(f"✅ Converted {converted_count} cross-references")
    return converted_count

def validate_internal_links(chapter_files: list) -> tuple[bool, list]:
    """Validate all internal cross-references.

    Args:
        chapter_files: List of chapter HTML file paths

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    log("🔍 Validating internal cross-references...")

    # Collect all anchors by file
    all_anchors = {}
    for html_file in chapter_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        anchors = {elem.get('id') for elem in soup.find_all(id=True) if elem.get('id')}
        all_anchors[html_file.name] = anchors
        log(f"  📍 {html_file.name}: {len(anchors)} anchors")

    # Validate all links
    broken = []
    total_links = 0

    for html_file in chapter_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link.get('href', '')

            # Skip external links
            if href.startswith(('http://', 'https://', '//', 'mailto:')):
                continue

            # Check internal anchors
            if '#' in href:
                total_links += 1
                if href.startswith('#'):
                    # Same-file anchor
                    anchor = href[1:]
                    if anchor not in all_anchors.get(html_file.name, set()):
                        broken.append(f"{html_file.name}: missing anchor '{anchor}'")
                else:
                    # Cross-file anchor
                    file_part, anchor = href.split('#', 1)
                    if file_part not in all_anchors:
                        broken.append(f"{html_file.name}: {href} → file '{file_part}' not found")
                    elif anchor not in all_anchors[file_part]:
                        broken.append(f"{html_file.name}: {href} → missing anchor '{anchor}' in {file_part}")

    if broken:
        log(f"❌ Found {len(broken)} broken link(s) out of {total_links} total:")
        for error in broken[:10]:  # Show first 10
            log(f"  • {error}")
        if len(broken) > 10:
            log(f"  ... and {len(broken) - 10} more")
        return False, broken
    else:
        log(f"✅ All {total_links} internal links validated successfully")
        return True, []

def validate_epub_with_epubcheck(epub_path: str) -> bool:
    """Run EPUBCheck validation on generated EPUB.

    Args:
        epub_path: Path to the EPUB file

    Returns:
        True if valid (or if epubcheck not available), False if validation failed
    """
    log("📚 Validating EPUB with EPUBCheck...")

    epubcheck_log = "epubcheck.log"

    try:
        # Check if epubcheck is available
        check_result = subprocess.run(
            ["epubcheck", "--version"],
            capture_output=True,
            text=True
        )

        if check_result.returncode != 0:
            log("⚠️  EPUBCheck not found (optional)")
            log("   Install with: brew install epubcheck")
            return True  # Don't fail build

        version = check_result.stdout.strip().split('\n')[0]
        log(f"   Using {version}")

        # Run validation
        result = subprocess.run(
            ["epubcheck", epub_path],
            capture_output=True,
            text=True
        )

        # Save full output to log file
        with open(epubcheck_log, 'w', encoding='utf-8') as f:
            f.write(f"EPUBCheck validation for: {epub_path}\n")
            f.write(f"Version: {version}\n")
            f.write("=" * 80 + "\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write("\n\nReturn code: " + str(result.returncode) + "\n")

        # Parse results
        if "No errors or warnings detected" in result.stdout:
            log("✅ EPUB validation passed")
            log(f"   Full report saved to: {epubcheck_log}")
            return True
        elif result.returncode == 0:
            log("⚠️  EPUB has warnings:")
            for line in result.stdout.split('\n'):
                if 'WARNING' in line or 'ERROR' in line:
                    log(f"   {line}")
            log(f"   Full report saved to: {epubcheck_log}")
            return True
        else:
            log(f"❌ EPUB validation failed:")
            for line in result.stdout.split('\n')[:20]:  # First 20 lines
                if line.strip():
                    log(f"   {line}")
            log(f"   Full report saved to: {epubcheck_log}")
            return False

    except FileNotFoundError:
        log("⚠️  EPUBCheck not installed (optional)")
        return True
    except Exception as e:
        log(f"⚠️  EPUBCheck validation error: {e}")
        return True  # Don't fail build

def unwrap_semantic_elements(html_dir: Path) -> bool:
    """Unwrap semantic HTML elements (article, main, nav, section) for ebook compatibility.

    Many ebook publishers don't support semantic HTML5 elements properly.
    This function unwraps these elements, keeping their content but removing the wrapper tags.
    """
    log("🔧 Unwrapping semantic HTML elements for ebook compatibility...")

    semantic_tags = ['article', 'main', 'nav', 'section']
    html_files = list(html_dir.glob("*.html"))

    if not html_files:
        log("⚠️  No HTML files found to process")
        return False

    total_unwrapped = 0

    for html_file in html_files:
        try:
            # Read the HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Count elements before unwrapping
            file_unwrapped = 0
            for tag in semantic_tags:
                elements = soup.find_all(tag)
                file_unwrapped += len(elements)
                for element in elements:
                    element.unwrap()

            if file_unwrapped > 0:
                # Write the modified HTML back
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(str(soup))

                total_unwrapped += file_unwrapped
                log(f"   ✓ {html_file.name}: unwrapped {file_unwrapped} semantic elements")

        except Exception as e:
            log(f"   ⚠️  Warning: Could not process {html_file.name}: {e}")

    if total_unwrapped > 0:
        log(f"✅ Unwrapped {total_unwrapped} semantic elements across {len(html_files)} files")
    else:
        log(f"✓ No semantic elements found (files already compatible)")

    return True

def convert_to_epub(metadata: dict, toc_file: Path, chapter_files: list, output_epub: str, epub_input_dir: Path = None) -> bool:
    """Convert chapter HTML files to EPUB using Calibre's ebook-convert.

    Args:
        metadata: EPUB metadata dictionary
        toc_file: Path to the table of contents HTML file (entry point)
        chapter_files: List of chapter HTML file paths
        output_epub: Output EPUB file path

    Returns:
        True if conversion successful, False otherwise
    """
    log("📖 Converting to EPUB with Calibre...")

    # Use TOC file as the main input
    # Calibre will follow links to all chapter files
    if not toc_file or not toc_file.exists():
        log(f"❌ TOC HTML file not found")
        return False

    # Resolve cover image path to absolute before changing directories
    cover_path_absolute = None
    if metadata.get('cover-image'):
        cover_path = Path(metadata['cover-image'])
        if not cover_path.is_absolute():
            # Resolve relative to metadata file location
            cover_path = Path(METADATA_FILE).parent / cover_path
        # Make it absolute by resolving it
        cover_path_absolute = cover_path.resolve()

    # Change to the directory containing the HTML files so Calibre can find them
    import os
    original_dir = os.getcwd()

    # Prepare paths based on whether we're changing directories
    if epub_input_dir:
        os.chdir(epub_input_dir)
        log(f"   Working directory: {epub_input_dir}")
        # Use just the filename when in the chapter directory
        input_file = toc_file.name
        # Make output path absolute relative to original directory
        output_path = str(Path(original_dir) / output_epub)
    else:
        input_file = str(toc_file)
        output_path = output_epub

    log(f"   Using {toc_file.name} as entry point")
    log(f"   Total chapters: {len(chapter_files)}")

    # Build ebook-convert command
    cmd = [
        EBOOK_CONVERT_COMMAND,
        input_file,
        output_path
    ]

    # Add metadata arguments
    if metadata.get('title'):
        title = metadata['title']
        if metadata.get('subtitle'):
            title = f"{title}: {metadata['subtitle']}"
        cmd.extend(["--title", title])
        log(f"   Title: {title}")

    if metadata.get('author'):
        # Handle both single author string and list of authors
        authors = metadata['author']
        if isinstance(authors, list):
            authors = ' & '.join(authors)
        cmd.extend(["--authors", authors])
        log(f"   Author(s): {authors}")

    if metadata.get('language'):
        cmd.extend(["--language", metadata['language']])

    if metadata.get('publisher'):
        cmd.extend(["--publisher", metadata['publisher']])

    if metadata.get('description'):
        cmd.extend(["--comments", metadata['description']])

    if metadata.get('isbn'):
        cmd.extend(["--isbn", metadata['isbn']])

    if metadata.get('rights'):
        cmd.extend(["--book-producer", metadata['rights']])

    if metadata.get('series'):
        cmd.extend(["--series", metadata['series']])
        if metadata.get('series-number'):
            cmd.extend(["--series-index", str(metadata['series-number'])])

    # Add cover image if specified and exists
    if cover_path_absolute:
        if cover_path_absolute.exists():
            cmd.extend(["--cover", str(cover_path_absolute)])
            log(f"   Cover image: {cover_path_absolute.name}")
        else:
            log(f"⚠️  Warning: Cover image not found: {cover_path_absolute.name}")

    # EPUB-specific options
    cmd.extend([
        "--no-default-epub-cover",  # Don't generate a default cover
        "--preserve-cover-aspect-ratio",  # Keep cover proportions
        "--flow-size", "0",  # Don't split HTML files by size
        "--insert-blank-line",  # Improve readability
        "--page-breaks-before", "/",  # No automatic page breaks
        # Add CSS to preserve TOC classes (prevents Calibre from stripping them)
        "--extra-css", "h1.part-title, h1.chapter-title, h1.appendix-title { display: block; } pre.codeblock, pre.codeblock code { font-family: monospace; white-space: pre-wrap; font-size: 0.85em; } code.ph.codeph { font-family: monospace; font-size: 0.85em; }",
        # Generate EPUB TOC from our hierarchical structure
        # Use contains() to match classes since h1 tags may have multiple space-separated classes
        "--level1-toc", "//h:h1[contains(@class, 'part-title') or contains(@class, 'appendix-title')]",  # Parts and Appendices
        "--level2-toc", "//h:h1[contains(@class, 'chapter-title')]",  # Regular chapters from actual content
        "--level3-toc", "//h:h1[@class='no-match-xyz']",  # Prevent deeper nesting
    ])

    log(f"   Converting {toc_file.name} to EPUB...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            log(f"✅ EPUB generated: {Path(output_path).name} ({file_size:.1f} KB)")
            return True
        else:
            log(f"❌ EPUB file not created at: {output_path}")
            return False

    except subprocess.CalledProcessError as e:
        log(f"❌ Error converting to EPUB:")
        if e.stdout:
            log(e.stdout)
        if e.stderr:
            log(e.stderr)
        return False
    finally:
        if epub_input_dir:
            os.chdir(original_dir)

# ---------------------------------------------------------------------
def main():
    # Clear old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log("🚀 Starting generate-epub.py (Calibre version)")
    log(f"📂 Reading DITA files from: {DITA_DIR}/")

    # Check prerequisites
    if not check_command(DITA_COMMAND, "Install from: https://www.dita-ot.org/download"):
        sys.exit(1)

    # Check for Calibre's ebook-convert
    if not os.path.exists(EBOOK_CONVERT_COMMAND):
        log(f"❌ Error: Calibre's ebook-convert not found at: {EBOOK_CONVERT_COMMAND}")
        log("   Install Calibre from: https://calibre-ebook.com/download")
        sys.exit(1)
    else:
        log(f"✓ ebook-convert found: {EBOOK_CONVERT_COMMAND}")

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

    # Step 1: Load EPUB metadata
    metadata_file = Path(METADATA_FILE)
    metadata = load_metadata(metadata_file)
    if not metadata:
        log(f"❌ Failed to load metadata")
        sys.exit(1)

    # Determine output EPUB filename from metadata
    output_epub = f"{metadata.get('title', 'Building AI Coding Assistants')}.epub"
    log(f"📖 Output EPUB: {output_epub}")

    # Step 2: Generate HTML5 with syntax highlighting
    html_output_dir = Path(HTML_OUTPUT_DIR)
    if not generate_html5(ditamap_path, html_output_dir):
        log(f"❌ Failed to generate HTML5")
        sys.exit(1)

    # Step 2.5: Rename HTML files to match original markdown names
    rename_map = rename_html_files_to_original_names(html_output_dir, dita_dir)

    # Step 3: Parse bookmap structure
    chapter_structure = parse_bookmap_structure(ditamap_path)
    if not chapter_structure:
        log(f"❌ Failed to parse bookmap structure")
        sys.exit(1)

    # Step 4: Generate chapter-based HTML files
    chapter_output_dir = Path(HTML_OUTPUT_DIR) / "chapters"
    chapter_files, topic_to_chapter = merge_html_by_chapter(
        chapter_structure,
        html_output_dir,
        chapter_output_dir
    )
    if not chapter_files:
        log(f"❌ Failed to generate chapter HTML files")
        sys.exit(1)

    # Step 4.5: Fix fragment-only anchor links (add topic prefix)
    fixed_anchors = fix_fragment_anchors(chapter_files)

    # Step 4.6: Fix EPUB validation errors (blockquotes, empty bodies)
    fix_epub_validation_errors(chapter_files)

    # Step 5: Convert cross-references
    converted_links = convert_cross_references(chapter_files, topic_to_chapter, chapter_structure)
    log(f"   Converted {converted_links} cross-reference links")

    # Step 6: Validate internal links
    links_valid, broken_links = validate_internal_links(chapter_files)
    if not links_valid:
        log(f"⚠️  Warning: {len(broken_links)} broken internal links found")
        log(f"   EPUB generation will continue, but some links may not work")

    # Step 7: Create table of contents HTML with nested structure
    toc_file = create_toc_html(chapter_structure, chapter_files, chapter_output_dir, metadata, html_output_dir)

    # Step 8: Convert to EPUB using Calibre
    if not convert_to_epub(metadata, toc_file, chapter_files, output_epub, chapter_output_dir):
        log(f"❌ Failed to create EPUB")
        sys.exit(1)

    # Step 9: Validate EPUB with EPUBCheck
    epub_valid = validate_epub_with_epubcheck(output_epub)
    if not epub_valid:
        log(f"⚠️  Warning: EPUB validation failed (see errors above)")

    log(f"🎉 Done! EPUB saved to {output_epub}")
    log("")
    log("📖 To view the EPUB:")
    log(f"   • macOS: open '{output_epub}'")
    log(f"   • Linux: ebook-viewer '{output_epub}'")
    log("   • Windows: Open with Calibre or Edge browser")
    log("")
    log("✨ Chapter-based structure for better navigation!")
    log(f"   To customize metadata, edit: {METADATA_FILE}")

if __name__ == "__main__":
    main()
