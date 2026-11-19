"""Document ingestion: validate -> extract text -> clean -> chunk.

Chunking strategy
-----------------
* Text is extracted page by page so every chunk keeps its page number for citations.
* Pages are split into sentences, and sentences are packed into chunks of up to
  `chunk_size` characters (~900 chars ≈ 200 tokens ≈ 1–2 paragraphs).
* Consecutive chunks overlap by roughly `chunk_overlap` characters (whole sentences),
  so a fact that straddles a boundary still appears intact in at least one chunk.
* Chunks never cross page boundaries; a very short trailing chunk is merged into the
  previous one so we don't index fragments with too little meaning to match well.

Why ~900 characters? Small enough that one chunk is about one idea (precise retrieval
and focused citations), large enough to carry context (definitions, the sentence before
a claim). With top_k = 5 the model sees ~4.5k characters — enough to answer, cheap to send.
"""
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

ALLOWED = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".markdown": "md"}


class IngestError(Exception):
    """A user-facing problem with an uploaded file."""


@dataclass
class Page:
    number: int
    text: str


@dataclass
class Chunk:
    page: int
    index: int
    text: str


# ---------------- extraction ----------------

def detect_kind(filename: str) -> str:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:
        raise IngestError(f"Unsupported file type '{ext or 'none'}'. Upload a PDF, TXT or Markdown file.")
    return ALLOWED[ext]


def extract_pages(data: bytes, kind: str) -> list[Page]:
    if not data:
        raise IngestError("The file is empty.")
    if kind == "pdf":
        return _extract_pdf(data)
    return _extract_text(data, kind)


def _extract_pdf(data: bytes) -> list[Page]:
    if not data.startswith(b"%PDF"):
        raise IngestError("This file is not a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise IngestError("This PDF is password-protected.") from e
        pages = [Page(i + 1, clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)]
    except IngestError:
        raise
    except (PdfReadError, Exception) as e:
        raise IngestError("Could not read this PDF - it may be corrupted.") from e

    pages = [p for p in pages if p.text]
    if not pages:
        raise IngestError("No text found in this PDF. It is probably a scanned image - "
                          "run it through OCR first or upload a text-based PDF.")
    return pages


def _extract_text(data: bytes, kind: str) -> list[Page]:
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
    if kind == "md":
        text = strip_markdown(text)
    text = clean(text)
    if len(text) < 20:
        raise IngestError("The file has too little text to index.")
    return [Page(1, text)]


# ---------------- cleaning ----------------

def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", lambda m: m.group(0).strip("`"), text, flags=re.S)  # keep code content
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)      # images -> alt text
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # links -> link text
    text = re.sub(r"^#{1,6}\s*(.+?)\s*#*$", r"\n\n\1\n\n", text, flags=re.M)  # headings -> own paragraph
    text = re.sub(r"(\*\*|__|\*|_|`)", "", text)                # emphasis / inline code
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.M)      # bullets
    return text


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n|• |\d+[.)] )", " ", text)  # unwrap hard-wrapped lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------- chunking ----------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[•])|\n{2,}|\n(?=• )")


def split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text)]
    out: list[str] = []
    for s in parts:
        if not s:
            continue
        # Hard-split pathological "sentences" (tables, missing punctuation).
        while len(s) > 1200:
            cut = s.rfind(" ", 0, 1000)
            cut = cut if cut > 200 else 1000
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        out.append(s)
    return out


def chunk_pages(pages: list[Page], size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        sentences = split_sentences(page.text)
        current: list[str] = []
        carried = 0          # how many sentences at the start of `current` are overlap
        length = 0
        page_chunks: list[str] = []
        for sent in sentences:
            if len(current) > carried and length + len(sent) + 1 > size:
                page_chunks.append(" ".join(current))
                # carry over trailing sentences as overlap
                carry: list[str] = []
                carry_len = 0
                for prev in reversed(current):
                    if carry_len + len(prev) > overlap:
                        break
                    carry.insert(0, prev)
                    carry_len += len(prev) + 1
                current, length, carried = carry, carry_len, len(carry)
            current.append(sent)
            length += len(sent) + 1
        new = current[carried:]
        if new:
            if page_chunks and len(" ".join(new)) < size * 0.25:
                page_chunks[-1] += " " + " ".join(new)   # merge tiny tail (without the overlap)
            else:
                page_chunks.append(" ".join(current))
        for text in page_chunks:
            chunks.append(Chunk(page=page.number, index=len(chunks), text=text))
    return chunks
