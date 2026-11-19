"""Parse PDF / TXT / Markdown into cleaned text blocks with page and section metadata.

Output is a flat list of `Block(page, section, text)`. A new block starts on every page
and on every detected heading, so chunks never mix two sections or two pages.

Section detection
-----------------
* Markdown: `#` headings, tracked as a path ("Assessment › Late policy").
* PDF / TXT: conservative patterns for headings students' material actually uses -
  "Q8. Explain ...", "Unit 3", "Chapter 2", "Lecture 5: ...", "2.1 Title" - on short lines.
  The current section carries over to following pages until a new heading appears.
"""
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

ALLOWED = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".markdown": "md"}


class IngestError(Exception):
    """A problem with an uploaded file that the user can act on."""


@dataclass
class Block:
    page: int
    section: str | None
    text: str


def detect_kind(filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:
        raise IngestError(f"Unsupported file type '{ext or 'none'}'. Upload a PDF, TXT or Markdown file.")
    return ALLOWED[ext]


def parse(data: bytes, kind: str) -> tuple[list[Block], int]:
    """Return (blocks, page_count)."""
    if not data:
        raise IngestError("The file is empty.")
    if kind == "pdf":
        pages = _pdf_pages(data)
        return _blocks_from_pages(pages), len(pages)
    text = _decode(data)
    if kind == "md":
        return _markdown_blocks(text), 1
    return _blocks_from_pages([text]), 1


# ---------------- extraction ----------------

def _pdf_pages(data: bytes) -> list[str]:
    if not data.startswith(b"%PDF"):
        raise IngestError("This file is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise IngestError("This PDF is password-protected.") from e
        pages = [p.extract_text() or "" for p in reader.pages]
    except IngestError:
        raise
    except Exception as e:
        raise IngestError("Could not read this PDF - it may be corrupted.") from e
    if not any(p.strip() for p in pages):
        raise IngestError("No text found in this PDF. It is probably a scanned image - "
                          "run it through OCR first or upload a text-based PDF.")
    return strip_repeated_lines(pages)


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise IngestError("Could not decode this text file. Save it as UTF-8 and try again.")
    if "\x00" in text:
        raise IngestError("This looks like a binary file, not text.")
    if len(text.strip()) < 20:
        raise IngestError("The file has too little text to index.")
    return text


# ---------------- headers / footers ----------------

def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.split())).lower()


def strip_repeated_lines(pages: list[str], edge: int = 3, min_share: float = 0.6) -> list[str]:
    """Remove running headers/footers (lines near the page edge that repeat on most pages,
    ignoring digits). They would otherwise be embedded into every chunk."""
    if len(pages) < 3:
        return pages
    split = [p.splitlines() for p in pages]
    counts: dict[str, int] = {}
    for lines in split:
        for key in {_line_key(l) for l in lines[:edge] + lines[-edge:] if l.strip()}:
            counts[key] = counts.get(key, 0) + 1
    repeated = {k for k, n in counts.items() if n >= max(3, min_share * len(pages)) and len(k) < 200}
    if not repeated:
        return pages
    out = []
    for lines in split:
        keep = [l for i, l in enumerate(lines)
                if not ((i < edge or i >= len(lines) - edge) and _line_key(l) in repeated)]
        out.append("\n".join(keep) if any(l.strip() for l in keep) else "\n".join(lines))
    return out


# ---------------- sections ----------------

_HEADING = re.compile(
    r"^\s*(?:"
    r"Q\.?\s?\d{1,3}[.):]\s+\S"                              # Q8. Explain ...
    r"|(?:Unit|Chapter|Lecture|Module|Section|Part)\s+[\dIVXLC]+\b"  # Unit 3 / Chapter IV
    r"|\d{1,2}(?:\.\d{1,2}){0,2}\s+[A-Z][^.!?]{2,80}$"       # 2.1 Scheduling Criteria
    r")"
)


def is_heading(line: str) -> bool:
    line = line.strip()
    return 3 <= len(line) <= 140 and bool(_HEADING.match(line))


def _blocks_from_pages(pages: list[str]) -> list[Block]:
    blocks: list[Block] = []
    section: str | None = None
    for page_no, raw in enumerate(pages, start=1):
        buf: list[str] = []
        for line in raw.splitlines():
            if is_heading(line):
                if buf:
                    blocks.append(Block(page_no, section, clean("\n".join(buf))))
                    buf = []
                section = " ".join(line.split())[:120]
            buf.append(line)
        if buf:
            blocks.append(Block(page_no, section, clean("\n".join(buf))))
    return [b for b in blocks if b.text]


def _markdown_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = clean(strip_markdown("\n".join(buf)))
        if body:
            blocks.append(Block(1, " › ".join(t for _, t in path) or None, body))
        buf.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if m:
            flush()
            level, title = len(m.group(1)), strip_markdown(m.group(2)).strip()
            path = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
            buf.append(title + ".")        # keep the heading words searchable
            continue
        buf.append(line)
    flush()
    return blocks


# ---------------- cleaning ----------------

def strip_markdown(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9]*\n?", "", text)                 # code fences (keep code)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)          # images -> alt text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)           # links -> text
    text = re.sub(r"(\*\*|__|\*|`)", "", text)                      # emphasis / inline code
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.M)          # bullets
    text = re.sub(r"^\s*\|?(\s*:?-{3,}:?\s*\|)+\s*$", "", text, flags=re.M)  # table rules
    return text


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n|• |\d+[.)] )", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
