# AGENTS.md — Technical Writing Agent for "Inside Unsloth"

This document describes how the AI agent acted as a **technical writer, editor, and proofreader** to produce a 40-chapter technical book documenting the Unsloth codebase. It captures the workflow, conventions, and lessons learned so that future agents can continue or adapt the process.

---

## Role Definition

The agent operated in three distinct roles across the project lifecycle:

1. **Technical Writer** — Researched source code, designed the content outline, and drafted all 40 chapters from scratch.
2. **Editor** — Deepened chapters with code walkthroughs, conceptual background, and expanded explanations.
3. **Proofreader** — Performed automated consistency checks and fixed structural, referential, and numbering errors.

---

## Project Structure

```
book/
├── metadata.yaml          # EPUB/PDF metadata (title, author, etc.)
├── md/
│   ├── toc.md             # Table of contents with ✅ completion markers
│   ├── EXAMPLE-chapter.md  # Template all chapters follow
│   └── chapter-NN-*.md     # 40 chapter files
├── generate-epub.py       # EPUB generation script
└── generate-dita.py       # DITA/PDF generation script
```

---

## Phase 1: Research and Outlining

### How the Outline Was Built

1. **Repository scan** — Used `list_dir`, `find_by_name`, and `wc -l` to map every file in the codebase, grouping by directory (`kernels/`, `models/`, `studio/`, etc.).
2. **Weight by complexity** — Larger files (e.g., `rl.py` at 82K, `import_fixes.py` at 65K) were flagged for deeper coverage. Small utility files were grouped into shared chapters.
3. **Dependency ordering** — Chapters were sequenced so that foundational concepts (installation, device detection, model loading) precede chapters that depend on them (training, saving, kernels).
4. **Nine-part structure** — Chapters were grouped into thematic parts (Orientation → Installation → Running → Training → Saving → Kernels → Architectures → Studio → Advanced Topics).

### Template Compliance

Every chapter follows the `EXAMPLE-chapter.md` template:

```markdown
# Chapter N: Title

> *"Epigraph — a memorable one-liner."*

---

## Introduction
[2–3 paragraphs of context]

### What You'll Learn
- Bullet list of learning objectives

### Prerequisites
- References to prior chapters

---

## N.1 First Section
...

## Source File Map
| Concept | Primary File(s) |
|---------|-----------------|
```

**Key rule**: Section numbers must match the chapter number (e.g., Chapter 23 has sections `23.1`, `23.2`, etc.). This was validated programmatically during the editorial pass.

---

## Phase 2: Writing Chapters

### Batching Strategy

The 40 chapters were written in **9 batches** of 3–8 chapters each, grouped by theme:

| Batch | Chapters | Theme |
|-------|----------|-------|
| 1 | 1–6 | Orientation & Installation |
| 2 | 7–11 | Running Models |
| 3 | 12–16 | Training (LoRA, RL, Vision) |
| 4 | 17–21 | Embedding, Data, Saving |
| 5 | 22–25 | Kernel Internals (Part 1) |
| 6 | 26–29 | Kernel Internals (Part 2) |
| 7 | 30–33 | Model Architectures |
| 8 | 34–37 | Unsloth Studio |
| 9 | 38–40 | Advanced Topics |

### Source Research Process

For each chapter, the agent followed this sequence:

1. **Read the primary source file(s)** using `view_file` — understanding the code before writing about it.
2. **Read the chapter notes** (the initial outline) to understand intended scope.
3. **Write the expanded chapter** — converting terse notes into prose with code examples, tables, and diagrams.
4. **Verify line count** — every chapter must be ≥150 lines. Chapters that fell short received additional sections (hardware requirements, evaluation, benchmarks).

### Writing Conventions

- **Code blocks** use the language's syntax highlighting (```python, ```bash, etc.)
- **ASCII diagrams** are used for architecture flows and data pipelines (no external image dependencies)
- **Tables** are used for comparisons, configuration parameters, and feature matrices
- **Epigraphs** are short, memorable quotes that capture the chapter's essence
- **Cross-references** use the format "Chapter N" (never hyperlinks between chapters, since the rendering format may vary)

---

## Phase 3: Deepening with Code Walkthroughs

### When a Chapter Needs Deepening

After the initial writing pass, chapters were reviewed for:
- **Line count** — chapters closer to the 150-line minimum were prioritized
- **Technical importance** — core kernel chapters (cross-entropy, RoPE, SwiGLU) benefit most from actual source code excerpts
- **Conceptual gaps** — algorithm chapters (LoRA, GRPO) benefit from mathematical derivations

### Code Walkthrough Format

Actual source code excerpts were annotated inline with comments explaining each line:

```python
# From _cross_entropy_forward (annotated):
c = tl.max(logits, 0)                               # c = max(logits)
logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))  # Stable LSE
```

**Key rule**: Walkthroughs show the *actual* kernel code (simplified where necessary for clarity), not pseudocode. This grounds the chapter in the real implementation and makes it useful as a reference.

### Chapters That Received Code Walkthroughs

| Chapter | Source File | What Was Added |
|---------|------------|---------------|
| Ch 23: Cross-Entropy | `cross_entropy_loss.py` | Forward kernel, chunked variant, backward kernel |
| Ch 25: RoPE | `rope_embedding.py` | Rotation kernel with grouped heads |
| Ch 26: SwiGLU | `swiglu.py` | Forward kernel, backward with buffer reuse |
| Ch 27: RMSNorm | `rms_layernorm.py` | Forward kernel, Gemma +1 variant |

### Chapters That Received Conceptual Background

| Chapter | What Was Added |
|---------|---------------|
| Ch 12: LoRA | SVD derivation explaining why low-rank updates work |
| Ch 15: GRPO | Formal algorithm pseudocode, advantage computation, reward function examples |
| Ch 24: Fast LoRA | NF4 quantization table, block-wise dequantization walkthrough |

---

## Phase 4: Editorial Pass

### Automated Checks

The editorial pass used shell scripts to verify consistency across all 40 chapters:

```bash
# Check structural elements:
for f in book/md/chapter-*.md; do
    grep -q '> *"' "$f"        || echo "Missing epigraph: $f"
    grep -q '## Introduction' "$f"  || echo "Missing intro: $f"
    grep -q 'Prerequisites' "$f"    || echo "Missing prereqs: $f"
    grep -q 'Source File Map' "$f"  || echo "Missing source map: $f"
done

# Check for duplicate section numbers:
for f in book/md/chapter-*.md; do
    chnum=$(basename "$f" | sed 's/chapter-0*//; s/-.*//')
    dups=$(grep -oE "^## ${chnum}\.[0-9]+" "$f" | sort | uniq -d)
    [ -n "$dups" ] && echo "DUPLICATE in $f: $dups"
done

# Verify all referenced source files exist:
for f in $(grep -roh 'unsloth/[a-z_/]*\.py' book/md/ | sort -u); do
    [ -f "$f" ] || echo "MISSING: $f"
done
```

### Issues Found and Fixed

| Issue | Where | Fix |
|-------|-------|-----|
| Duplicate section `## 23.3` | Ch 23 | Renumbered sections 23.3–23.6 → 23.4–23.7 |
| Nonexistent `unsloth/models/deepseek.py` | Ch 29, 33 | Corrected to `unsloth/registry/_deepseek.py` |

---

## Lessons Learned

### What Worked Well

1. **Read the source first, write second** — Reading the actual Triton kernels before writing produced far more accurate and useful chapters than working from file names alone.
2. **Batch by theme, not by number** — Grouping related chapters (e.g., all kernel chapters) allowed the agent to build context once and reuse it across multiple chapters.
3. **Minimum line count enforcement** — The ≥150 line requirement prevented shallow chapters and forced the agent to add meaningful detail rather than stopping at surface descriptions.
4. **Automated structural checks** — Running `grep` checks across all chapters caught errors that manual review would miss (duplicate section numbers, broken file references).
5. **Template compliance** — Having one `EXAMPLE-chapter.md` as the standard eliminated formatting inconsistency across the nine writing batches.

### What to Watch For

1. **Section numbering after insertions** — When inserting a new section (e.g., a code walkthrough between §23.2 and §23.3), all subsequent sections must be renumbered. This was the most common error type.
2. **File path accuracy** — Source file references in the "Source File Map" table must be verified against the actual file system. Files move or get renamed — don't trust notes from earlier research.
3. **Merge artifact corruption** — When using `replace_file_content` with imprecise `TargetContent`, the tool can produce garbled output. Always verify the file after edits.
4. **macOS grep lacks `-P`** — Use `grep -E` (extended regex) instead of `grep -P` (Perl regex) on macOS. This tripped the agent during the editorial pass.

### Style Guide Summary

| Element | Convention |
|---------|-----------|
| Chapter title | `# Chapter N: Title` |
| Epigraph | `> *"Quote."*` (italicized, in quotes) |
| Section headers | `## N.X Section Name` (match chapter number) |
| Code blocks | ` ```python ` with syntax highlighting |
| Tables | GitHub-flavored Markdown pipe tables |
| Cross-references | "Chapter N" in prose (no hyperlinks) |
| Source maps | Table with `| Concept | Primary File(s) |` at chapter end |
| Horizontal rules | `---` before each major section and before Source File Map |

---

## Final Statistics

- **40 chapters** across 9 thematic parts
- **7,698 lines** of publication-ready prose
- **All chapters ≥ 150 lines**; longest is 283 lines (Ch 2: Repository Tour)
- **All referenced source files verified** to exist on disk
- **Zero duplicate section numbers** across the entire book
