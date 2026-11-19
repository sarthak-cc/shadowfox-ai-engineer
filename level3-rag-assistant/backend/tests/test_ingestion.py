import pytest

from app.ingestion.chunking import Chunk, chunk_blocks
from app.ingestion.parsers import Block, IngestError, detect_kind, is_heading, parse, strip_repeated_lines

from .conftest import SAMPLES


def test_pdf_pages_and_sections():
    blocks, pages = parse((SAMPLES / "os-lecture-cpu-scheduling.pdf").read_bytes(), "pdf")
    assert pages == 3
    assert {b.page for b in blocks} == {1, 2, 3}
    assert blocks[0].section == "Lecture 5: CPU Scheduling"


def test_markdown_sections_form_a_path():
    md = b"# Handbook\n\nIntro text that is long enough.\n\n## Late policy\n\nLate work loses **10%** per day. See [rules](http://x.y)."
    blocks, _ = parse(md, "md")
    late = [b for b in blocks if "10%" in b.text][0]
    assert late.section == "Handbook › Late policy"
    assert "**" not in late.text and "http" not in late.text


@pytest.mark.parametrize("line, expected", [
    ("Q8. Explain Declarative and Scripted Pipelines.", True),
    ("Unit III: Continuous Integration", True),
    ("2.1 Scheduling Criteria", True),
    ("The quick brown fox jumps over the lazy dog.", False),
    ("3 apples.", False),
])
def test_heading_detection(line, expected):
    assert is_heading(line) is expected


def test_chunks_never_cross_sections_and_overlap():
    text = " ".join(f"Sentence {i} explains part {i} of the topic in some detail." for i in range(60))
    blocks = [Block(1, "A", text), Block(1, "B", text)]
    chunks = chunk_blocks(blocks, size=400, overlap=80)
    assert all(len(c.text) <= 400 * 1.3 for c in chunks)
    assert {c.section for c in chunks} == {"A", "B"}
    a = [c for c in chunks if c.section == "A"]
    assert a[0].text.split(". ")[-1].rstrip(".") in a[1].text


def test_contextual_header_used_for_embedding_only():
    c = Chunk(page=2, section="Late policy", text="It costs 10% per day.")
    assert c.embed_text("handbook.md").startswith("Document: handbook.md | Section: Late policy\n")
    assert c.text == "It costs 10% per day."


def test_running_headers_removed():
    pages = [f"Course X | Notes\nPage {i}\n{t} explained here.\nFooter" for i, t in
             enumerate(["Threads", "Paging", "Files", "Locks"], start=1)]
    out = strip_repeated_lines(pages)
    assert all("Course X" not in p and "Page" not in p and "Footer" not in p for p in out)


@pytest.mark.parametrize("name, data, fragment", [
    ("a.docx", b"x", "Unsupported"),
    ("a.pdf", b"not a pdf", "not a valid PDF"),
    ("a.txt", b"", "empty"),
    ("a.txt", b"tiny", "too little text"),
])
def test_bad_files(name, data, fragment):
    with pytest.raises(IngestError, match=fragment):
        parse(data, detect_kind(name))
