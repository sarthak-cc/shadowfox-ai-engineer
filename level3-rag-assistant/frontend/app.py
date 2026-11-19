"""Groundwork - Streamlit frontend for the RAG assistant API."""
import html
import re
import time

import streamlit as st

import api

st.set_page_config(page_title="Groundwork", page_icon=":material/menu_book:", layout="wide", initial_sidebar_state="expanded")

SAMPLE_QUESTIONS = {
    "course-handbook.md": ["What happens if I submit an assignment late?", "How is the final grade calculated?"],
    "os-lecture-cpu-scheduling.pdf": ["What is the convoy effect?", "How should the Round Robin time quantum be chosen?"],
}
GENERIC_QUESTIONS = ["Summarise the main topics of these documents", "What are the most important definitions?"]
STEP_LABELS = {
    "rewrite": "Understanding the question…", "retrieve": "Searching documents…", "rerank": "Ranking passages…",
    "select": "Selecting evidence…", "generate": "Writing the answer…", "check": "Checking every claim…",
    "revise": "Fixing unsupported claims…", "no_answer": "Nothing relevant found…", "finalize": "Finishing…",
}

# ------------------------------------------------------------------ styles
LIGHT = """:root { --ink:#1b1a17; --ink2:#4a463f; --muted:#8a8478; --line:#e5ddcf; --line2:#d4c9b6; --paper:#f7f4ee;
  --surface:#fffdf9; --side:#f3efe7; --input:#fffdf9; --chip:#efe9df; --chip2:#e8e2d6; --track:#ece6da;
  --accent:#223fd1; --on-accent:#ffffff; --brand:#223fd1; --accent-soft:#e6eaff; --hl:#fff1b8;
  --good:#137a4b; --good-soft:#def5e8; --warn:#a86a00; --warn-soft:#fff0cf; --bad:#c2341f; --bad-soft:#fde5df;
  color-scheme: light; }"""
DARK = """:root { --ink:#f3efe7; --ink2:#cdc6b8; --muted:#8f897c; --line:#2e2b26; --line2:#3d3932; --paper:#12110f;
  --surface:#1a1916; --side:#161512; --input:#1f1d1a; --chip:#26241f; --chip2:#2e2b26; --track:#2e2b26;
  --accent:#8fa2ff; --on-accent:#0d0f1f; --brand:#3d5afe; --accent-soft:#1f2442; --hl:#3a3214;
  --good:#5fd39a; --good-soft:#13291e; --warn:#f0b84a; --warn-soft:#33270e; --bad:#ff8a76; --bad-soft:#35170f;
  color-scheme: dark; }"""

CSS = """
.stApp, .stApp button, .stApp input, .stApp textarea { font-family: "Inter", system-ui, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: "Material Symbols Rounded" !important; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stAppViewContainer"] { background: var(--paper) radial-gradient(var(--line) 1px, transparent 1px) -11px -11px / 22px 22px; }
.block-container { max-width: 980px; padding-top: 2.2rem; padding-bottom: 7rem; }
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

.brand { display:flex; align-items:center; gap:10px; margin: 0; }
.brand .mark { width:34px; height:34px; border-radius:9px; background:var(--brand); display:grid; place-items:center;
               color:#ffd23f; font-size:18px; font-weight:700; }
.brand .name { font: 400 28px/1 "Instrument Serif", Georgia, serif; color:var(--ink); }
.brand .name em { color: var(--accent); }
.stApp .eyebrow { font: 500 11px/1 "JetBrains Mono", monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin: 1rem 0 .5rem; }
.hero h1 { font: 400 clamp(38px,4.4vw,56px)/1.04 "Instrument Serif", Georgia, serif !important; letter-spacing:-.015em; margin:.3rem 0 .6rem; padding:0; color: var(--ink); }
.hero h1 em { background: linear-gradient(transparent 62%, var(--hl) 62%); padding: 0 .06em; }
.stApp .hero p { color: var(--ink2); font-size: 16.5px; max-width: 62ch; }
.flow { display:flex; flex-wrap:wrap; gap:6px; margin: 1.1rem 0 1.6rem; }
.flow span { font: 500 11.5px "JetBrains Mono", monospace; padding:6px 10px; border-radius:99px; background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.flow i { color: var(--muted); font-style: normal; align-self:center; }

.status-dot { display:inline-flex; align-items:center; gap:7px; font-size:12.5px; color:var(--muted); }
.status-dot b { width:8px; height:8px; border-radius:50%; background:var(--good); box-shadow: 0 0 0 4px var(--good-soft); }
.status-dot.err b { background: var(--bad); box-shadow: 0 0 0 4px var(--bad-soft); }

.stApp .doc-meta { font-size: 12px; color: var(--muted); margin: -14px 0 8px 28px; }
.badge { font: 600 10px "JetBrains Mono", monospace; padding: 2px 6px; border-radius: 5px; margin-right: 4px; }
.badge.pdf { background: var(--bad-soft); color: var(--bad); } .badge.md { background: var(--accent-soft); color: var(--accent); }
.badge.txt { background: var(--good-soft); color: var(--good); }

.pills { display:flex; flex-wrap:wrap; gap:6px; margin: 2px 0 10px; }
.pill { display:inline-flex; align-items:center; gap:6px; font: 600 11px "JetBrains Mono", monospace; padding: 4px 9px; border-radius: 99px; background:var(--chip); color:var(--ink2); }
.pill.high, .pill.answered { background: var(--good-soft); color: var(--good); }
.pill.medium, .pill.revised, .pill.unverified { background: var(--warn-soft); color: var(--warn); }
.pill.low, .pill.none, .pill.not_found { background: var(--bad-soft); color: var(--bad); }
.cite { display:inline-block; min-width:19px; padding:0 5px; margin:0 1px; border-radius:6px; background:var(--accent-soft);
        color:var(--accent); font: 600 11px/19px "JetBrains Mono", monospace; text-align:center; vertical-align: 1px; }
.answer strong { background: linear-gradient(transparent 60%, var(--hl) 60%); }

.src { border:1px solid var(--line); border-radius: 12px; background: var(--surface); padding: 12px 14px; margin-bottom: 8px; }
.src.uncited { opacity: .7; }
.src-head { display:flex; align-items:center; gap:10px; font-size: 13.5px; margin-bottom: 6px; flex-wrap: wrap; }
.src-n { display:grid; place-items:center; width:22px; height:22px; border-radius:7px; background:var(--accent); color:var(--on-accent); font: 600 11px "JetBrains Mono", monospace; flex:none; }
.src.uncited .src-n { background:var(--chip2); color: var(--ink2); }
.stApp .src-head .where { color: var(--muted); }
.stApp .src-scores { margin-left:auto; display:flex; gap:10px; font: 500 11px "JetBrains Mono", monospace; color: var(--ink2); }
.bar { display:inline-block; width:46px; height:5px; border-radius:99px; background:var(--track); vertical-align: middle; margin-right: 4px; overflow:hidden; }
.bar i { display:block; height:100%; background: var(--good); }
.stApp .src-text { font-size: 13.8px; line-height: 1.55; color: var(--ink2); margin: 0; }
.stApp .claim { display:flex; gap:8px; font-size: 14px; margin: 4px 0; }
.claim .ok { color: var(--good); } .claim .no { color: var(--bad); }
.stApp .trace-q { font-size: 13px; color: var(--ink2); margin-bottom: 6px; }
.trace-q code { font-size: 12px; }
[data-testid="stChatMessage"] { background: transparent; }
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-child(2) button { padding: 0 !important; min-width: 32px; min-height: 32px; border-color: transparent; background: transparent; color: var(--muted); }
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-child(2) button:hover { color: var(--bad); background: var(--bad-soft); }
[data-testid="stSidebar"] [data-testid="stColumn"]:nth-child(2) .stMarkdownEllipsis { overflow: visible; }
/* ---- Streamlit widgets follow the same palette (needed for dark mode) ---- */
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] :where(div, b, strong, em) { color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.stApp :where(p, li, h1, h2, h3, h4, h5, label, summary, small, td, th) { color: var(--ink); }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], .stApp [data-testid="stMarkdownContainer"] small { color: var(--ink2); }
.stApp code { background: var(--chip); color: var(--ink); }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: var(--paper); }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] div:has(> textarea) { background: var(--input) !important; border-color: var(--line2) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--accent); color: var(--on-accent); }
[data-testid="stChatInputSubmitButton"]:disabled { background: var(--chip); color: var(--muted); }
[data-testid="stChatInput"] textarea { color: var(--ink); background: var(--input); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); opacity: 1; }
[data-testid="stChatInput"] textarea:disabled { -webkit-text-fill-color: var(--muted); cursor: not-allowed; }
.empty-card { padding: 16px 18px; border-radius: 12px; background: var(--accent-soft); border: 1px solid var(--line); color: var(--ink); font-size: 15px; }
.empty-card b { color: var(--ink); }
.st-key-theme_btn button { width: 38px; height: 38px; min-height: 38px; padding: 0 !important; border-radius: 50% !important; background: var(--surface) !important; border: 1px solid var(--line2) !important; color: var(--ink) !important; }
.st-key-theme_btn button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
.st-key-theme_btn button * { color: inherit !important; }
.st-key-theme_btn .stMarkdownEllipsis { overflow: visible; }
[data-testid="stFileUploaderDropzone"] { background: var(--surface); border: 1px dashed var(--line2); }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink2); }
[data-testid="stBaseButton-secondary"] { background: var(--surface); color: var(--ink); border-color: var(--line2); }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--accent); color: var(--accent); }
[data-testid="stBaseButton-primary"] { background: var(--accent); color: var(--on-accent); border-color: var(--accent); }
[data-testid="stExpander"] details { background: var(--surface); border-color: var(--line); border-radius: 12px; }
[data-testid="stExpander"] summary { background: transparent !important; color: var(--ink) !important; }
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:focus-visible { background: var(--chip) !important; }
[data-testid="stExpander"] summary:hover * { color: var(--accent); }
[data-testid="stExpander"] summary svg, [data-testid="stExpander"] [data-testid="stIconMaterial"] { color: var(--ink2); }
[data-testid^="stChatMessageAvatar"] { background: var(--surface) !important; border: 1px solid var(--line); }
[data-testid^="stChatMessageAvatar"] * { color: var(--accent) !important; }
[data-testid="stAlertContainer"] { background: var(--chip) !important; border: 1px solid var(--line); }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-baseweb="tooltip"], [data-baseweb="tooltip"] > div, [data-testid="stTooltipContent"] { background: var(--chip2) !important; color: var(--ink) !important; }
[data-baseweb="tooltip"] * { color: var(--ink) !important; }
[data-testid="stSidebar"] hr { border-color: var(--line); }
.trace { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
.trace th { text-align: left; font: 500 10.5px "JetBrains Mono", monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); padding: 6px 8px; border-bottom: 1px solid var(--line); }
.trace td { padding: 7px 8px; border-bottom: 1px solid var(--line); color: var(--ink2); vertical-align: top; }
.trace td:first-child { font: 600 12px "JetBrains Mono", monospace; color: var(--ink); white-space: nowrap; }
.trace td.ms { text-align: right; font: 500 12px "JetBrains Mono", monospace; white-space: nowrap; }
.trace .msbar { display:inline-block; height:4px; border-radius:2px; background: var(--accent); vertical-align: middle; margin-right: 6px; opacity: .7; }
"""
# One line, no blank lines: Streamlit's Markdown parser can't split the block and print CSS as text.
# The dark-mode toggle (sidebar) stores its value in session_state; read it before drawing anything.
_palette = DARK if st.session_state.get("dark", False) else LIGHT
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet"><style>' + " ".join((_palette + CSS).split()) + "</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(s: str) -> str:
    return html.escape(str(s or ""))


def render_answer_md(text: str) -> str:
    """Model output is untrusted: neutralise raw HTML, then turn [n] into citation chips."""
    safe = (text or "").replace("<", "&lt;")
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', safe)


@st.cache_data(ttl=8, show_spinner=False)
def get_health():
    try:
        return api.health()
    except api.ApiError as e:
        return {"error": str(e)}


def load_docs() -> list[dict]:
    try:
        return api.documents()
    except api.ApiError as e:
        st.sidebar.error(str(e))
        return []


ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("excluded", set())
ss.setdefault("uploader_key", 0)
ss.setdefault("pending_question", None)


# ------------------------------------------------------------------ sidebar
def _toggle_theme() -> None:
    st.session_state["dark"] = not st.session_state.get("dark", False)


with st.sidebar:
    b1, b2 = st.columns([0.8, 0.2], vertical_alignment="center")
    b1.markdown('<div class="brand"><div class="mark">◆</div><div class="name">Ground<em>work</em></div></div>',
                unsafe_allow_html=True)
    dark = st.session_state.get("dark", False)
    b2.button(":material/light_mode:" if dark else ":material/dark_mode:", key="theme_btn", on_click=_toggle_theme,
              help="Switch to light mode" if dark else "Switch to dark mode")
    h = get_health()
    if "error" in h:
        st.markdown(f'<span class="status-dot err"><b></b>Backend offline</span>', unsafe_allow_html=True)
        st.caption(h["error"])
    else:
        rr = "reranker ready" if h["reranker"] == "ready" else f"reranker {h['reranker']}"
        st.markdown(f'<span class="status-dot"><b></b>{esc(h["models"]["generation"][0])} · {esc(rr)}</span>',
                    unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">Add documents</p>', unsafe_allow_html=True)
    files = st.file_uploader("PDF, TXT or Markdown", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True,
                             key=f"up_{ss.uploader_key}", label_visibility="collapsed")
    if files and st.button(f"Index {len(files)} file{'s' if len(files) > 1 else ''}", type="primary", use_container_width=True):
        for f in files:
            box = st.empty()
            bar = st.progress(0.0, text=f"{f.name}: uploading…")
            try:
                j = api.upload(f.name, f.getvalue())
                while j["status"] == "processing":
                    if j["stage"] == "waiting":
                        label = f"{f.name}: free-tier limit, resuming in {int(j['wait_s'] or 0)}s"
                    elif j["stage"] == "embedding" and j["total"]:
                        label = f"{f.name}: embedding {j['done']}/{j['total']}"
                    else:
                        label = f"{f.name}: {j['stage']}…"
                    bar.progress(min(1.0, j["done"] / j["total"]) if j["total"] else 0.02, text=label)
                    time.sleep(0.6)
                    j = api.job(j["id"])
                if j["status"] == "error":
                    bar.empty()
                    box.error(f"{f.name}: {j['error']}")
                else:
                    d = j["document"]
                    bar.progress(1.0, text=f"{f.name}: {d['chunks']} chunks in {j['elapsed_s']}s")
            except api.ApiError as e:
                bar.empty()
                box.error(f"{f.name}: {e}")
        ss.uploader_key += 1
        time.sleep(0.8)
        st.rerun()

    docs = load_docs()
    st.markdown(f'<p class="eyebrow">Library · {len(docs)} file{"s" if len(docs) != 1 else ""}</p>', unsafe_allow_html=True)
    if not docs:
        st.caption("No documents yet.")
        if st.button("Load sample documents", use_container_width=True):
            with st.spinner("Indexing samples…"):
                try:
                    api.load_samples()
                except api.ApiError as e:
                    st.error(str(e))
            st.rerun()
    for d in docs:
        c1, c2 = st.columns([0.86, 0.14])
        on = c1.checkbox(d["name"], value=d["id"] not in ss.excluded, key=f"chk_{d['id']}")
        (ss.excluded.discard if on else ss.excluded.add)(d["id"])
        if c2.button(":material/close:", key=f"del_{d['id']}", help=f"Remove {d['name']}"):
            try:
                api.delete(d["id"])
            except api.ApiError as e:
                st.error(str(e))
            ss.excluded.discard(d["id"])
            st.rerun()
        sec = f" · {d['sections']} section{'s' if d['sections'] != 1 else ''}" if d.get("sections") else ""
        st.markdown(f'<div class="doc-meta"><span class="badge {d["kind"]}">{d["kind"].upper()}</span>'
                    f'{d["pages"]} p · {d["chunks"]} chunks{sec}</div>', unsafe_allow_html=True)

    st.markdown('<p class="eyebrow">Settings</p>', unsafe_allow_html=True)
    top_k = st.slider("Passages given to the model", 2, 10, 5)
    show_trace = st.toggle("Show pipeline details", value=True)
    if ss.messages and st.button("Clear conversation", use_container_width=True):
        ss.messages = []
        st.rerun()

selected = [d["id"] for d in docs if d["id"] not in ss.excluded]


# ------------------------------------------------------------------ rendering
def render_meta(r: dict) -> None:
    status_label = {"answered": "Answered", "not_found": "Not in documents", "revised": "Revised after check",
                    "unverified": "Not verified"}[r["status"]]
    pills = [f'<span class="pill {r["status"]}">{status_label}</span>']
    if r["status"] != "not_found":
        pills.append(f'<span class="pill {r["confidence"]}">{r["confidence"]} confidence · {r["confidence_score"]:.2f}</span>')
    g = r.get("groundedness")
    if g and g["claims"]:
        ok = sum(1 for c in g["claims"] if c["supported"])
        pills.append(f'<span class="pill">{ok}/{len(g["claims"])} claims supported</span>')
    pills.append(f'<span class="pill">{r["total_ms"] / 1000:.1f}s</span>')
    st.markdown(f'<div class="pills">{"".join(pills)}</div>', unsafe_allow_html=True)


def render_sources(r: dict) -> None:
    srcs = r["sources"]
    if not srcs:
        return
    title = "Sources" if r["status"] != "not_found" else "Closest passages (not enough to answer)"
    cited = sum(1 for s in srcs if s["cited"])
    with st.expander(f"{title} · {len(srcs)}" + (f" ({cited} cited)" if cited else ""), expanded=False):
        for s in srcs:
            rer = s.get("rerank_score")
            bar = f'<span class="bar"><i style="width:{max(4, int((rer or 0) * 100))}%"></i></span>' if rer is not None else ""
            ranks = " ".join(x for x in [
                f"dense #{s['dense_rank']}" if s.get("dense_rank") else "",
                f"bm25 #{s['bm25_rank']}" if s.get("bm25_rank") else ""] if x)
            section = f" · {esc(s['section'])}" if s.get("section") else ""
            st.markdown(f"""<div class="src {'' if s['cited'] or r['status'] == 'not_found' else 'uncited'}">
<div class="src-head"><span class="src-n">{s['n']}</span><b>{esc(s['document_name'])}</b>
<span class="where">p.{s['page']}{section}</span>
<span class="src-scores">{bar}{'relevance ' + format(rer, '.2f') if rer is not None else ''} <span>{ranks}</span></span></div>
<p class="src-text">{esc(s['text'])}</p></div>""", unsafe_allow_html=True)


def render_details(r: dict) -> None:
    g = r.get("groundedness")
    if g and g["claims"]:
        with st.expander(f"Claim check · {g['score'] * 100:.0f}% supported"):
            for c in g["claims"]:
                mark = '<span class="ok">✓</span>' if c["supported"] else '<span class="no">✗</span>'
                refs = "".join(f'<span class="cite">{n}</span>' for n in c["sources"])
                st.markdown(f'<div class="claim">{mark}<span>{esc(c["claim"])} {refs}</span></div>', unsafe_allow_html=True)
            if g["invalid_citations"]:
                st.caption(f"Citations to missing passages removed: {g['invalid_citations']}")
    if show_trace:
        with st.expander("Pipeline trace"):
            qs = " · ".join(f"<code>{esc(q)}</code>" for q in r["search_queries"])
            st.markdown(f'<div class="trace-q"><b>Searched as:</b> {esc(r["standalone_question"])}<br>'
                        f'<b>Queries:</b> {qs}</div>', unsafe_allow_html=True)
            longest = max([t["ms"] for t in r["trace"]] + [1])
            rows = "".join(
                f'<tr><td>{esc(t["node"])}</td><td class="ms"><span class="msbar" style="width:{max(2, int(60 * t["ms"] / longest))}px"></span>'
                f'{t["ms"]:,} ms</td><td>{esc(t["detail"])}</td></tr>' for t in r["trace"])
            st.markdown(f'<table class="trace"><tr><th>Step</th><th>Time</th><th>Detail</th></tr>{rows}</table>',
                        unsafe_allow_html=True)


def render_assistant(r: dict) -> None:
    render_meta(r)
    st.markdown(f'<div class="answer">\n\n{render_answer_md(r["answer"])}\n\n</div>', unsafe_allow_html=True)
    render_sources(r)
    render_details(r)


# ------------------------------------------------------------------ main area
if not ss.messages:
    st.markdown("""<div class="hero"><p class="eyebrow">Grounded document assistant</p>
<h1>Ask your files. <em>Check every answer.</em></h1>
<p>Answers come only from the documents in your library. Every claim is cited, then checked against the
passages it cites, and you can open each source to see exactly where it came from.</p></div>
<div class="flow"><span>rewrite</span><i>→</i><span>hybrid search</span><i>→</i><span>rerank</span><i>→</i>
<span>select evidence</span><i>→</i><span>stream answer</span><i>→</i><span>verify claims</span></div>""",
                unsafe_allow_html=True)
    if docs:
        qs = [q for d in docs for q in SAMPLE_QUESTIONS.get(d["name"], [])] or GENERIC_QUESTIONS
        st.markdown('<p class="eyebrow">Try asking</p>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(qs[:4]):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                ss.pending_question = q
                st.rerun()
    else:
        st.markdown('<div class="empty-card"><b>Your library is empty.</b> Add PDF, TXT or Markdown files in the '
                    'sidebar, or click <b>Load sample documents</b>, to get started.</div>', unsafe_allow_html=True)

for m in ss.messages:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/auto_awesome:"):
        if m["role"] == "user":
            st.markdown(esc(m["content"]))
        elif m.get("error"):
            st.error(m["error"])
        else:
            render_assistant(m["response"])

question = st.chat_input("Ask a question about your documents…" if docs else "Add documents in the sidebar to start asking…",
                         disabled=not docs)
if ss.pending_question:
    question, ss.pending_question = ss.pending_question, None

if question:
    if not selected:
        st.warning("Tick at least one document in the sidebar to search.")
        st.stop()
    history = [{"role": m["role"], "content": m["content"] if m["role"] == "user" else m["response"]["answer"]}
               for m in ss.messages if not m.get("error")][-8:]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(esc(question))

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        status = st.status("Understanding the question…", expanded=False)
        answer_box = st.empty()
        text, final = "", None
        try:
            for ev in api.chat_stream(question, history, None if len(selected) == len(docs) else selected, top_k):
                t = ev["type"]
                if t == "step" and ev["status"] == "start":
                    status.update(label=STEP_LABELS.get(ev["node"], ev["node"]))
                elif t == "step" and ev["status"] == "end":
                    status.write(f"**{ev['node']}** · {ev['ms']} ms · {ev.get('detail', '')}")
                elif t == "sources":
                    status.update(label=f"Writing the answer from {len(ev['sources'])} passages…")
                elif t == "token":
                    text += ev["text"]
                    answer_box.markdown(f'<div class="answer">\n\n{render_answer_md(text)} ▍\n\n</div>', unsafe_allow_html=True)
                elif t == "revision":
                    text = ev["answer"]
                    answer_box.markdown(f'<div class="answer">\n\n{render_answer_md(text)}\n\n</div>', unsafe_allow_html=True)
                elif t == "error":
                    raise api.ApiError(ev["error"])
                elif t == "done":
                    final = ev["response"]
            if final is None:
                raise api.ApiError("The answer stream ended unexpectedly. Please try again.")
            status.update(label=f"Done in {final['total_ms'] / 1000:.1f}s", state="complete")
            ss.messages.append({"role": "assistant", "content": final["answer"], "response": final})
        except api.ApiError as e:
            status.update(label="Something went wrong", state="error")
            ss.messages.append({"role": "assistant", "content": "", "error": str(e)})
        st.rerun()
