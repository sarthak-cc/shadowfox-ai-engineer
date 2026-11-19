"""Sentence-aware chunking with overlap, inside section/page blocks.

* Blocks (one page x one section) are split into sentences, which are packed into
  chunks of up to `size` characters. A chunk never ends mid-sentence.
* Consecutive chunks of a block share ~`overlap` characters of whole sentences.
* A tiny leftover is merged into the previous chunk of the same block.
* Chunks never cross a page or section boundary, so each citation points to one
  page and one section.

For embedding, each chunk is prefixed with a short context header
("Document: x | Section: y") - "contextual chunk headers". A passage like
"It has three stages..." then still matches a question about the section topic.
The stored/displayed text stays unprefixed.
"""
import re
from dataclasses import dataclass

from .parsers import Block

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[•])|\n{2,}|\n(?=• )")


@dataclass
class Chunk:
    page: int
    section: str | None
    text: str

    def embed_text(self, document_name: str) -> str:
        header = f"Document: {document_name}"
        if self.section:
            header += f" | Section: {self.section}"
        return f"{header}\n{self.text}"


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for s in (p.strip() for p in _SENTENCE_SPLIT.split(text)):
        if not s:
            continue
        while len(s) > 1500:          # tables / missing punctuation
            cut = s.rfind(" ", 0, 1200)
            cut = cut if cut > 300 else 1200
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        out.append(s)
    return out


def chunk_blocks(blocks: list[Block], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in blocks:
        parts: list[str] = []
        current: list[str] = []
        carried = 0
        length = 0
        for sent in split_sentences(block.text):
            if len(current) > carried and length + len(sent) + 1 > size:
                parts.append(" ".join(current))
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
            if parts and len(" ".join(new)) < size * 0.25:
                parts[-1] += " " + " ".join(new)
            else:
                parts.append(" ".join(current))
        chunks.extend(Chunk(block.page, block.section, p) for p in parts)
    return _merge_tiny(chunks, size)


def _merge_tiny(chunks: list[Chunk], size: int, min_chars: int = 120) -> list[Chunk]:
    """Merge very small chunks (e.g. a lone heading) into the next chunk of the same page."""
    out: list[Chunk] = []
    pending: Chunk | None = None
    for c in chunks:
        if pending is not None:
            if c.page == pending.page and len(pending.text) + len(c.text) <= size * 1.3:
                c = Chunk(c.page, c.section or pending.section, pending.text + " " + c.text)
            else:
                out.append(pending)
            pending = None
        if len(c.text) < min_chars:
            pending = c
        else:
            out.append(c)
    if pending is not None:
        if out and out[-1].page == pending.page:
            last = out[-1]
            out[-1] = Chunk(last.page, last.section, last.text + " " + pending.text)
        else:
            out.append(pending)
    return out
