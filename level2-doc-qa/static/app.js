/* DocLens frontend — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const MAX_MB = 15;
  const ALLOWED = ["pdf", "txt", "md", "markdown"];
  const SAMPLE_QUESTIONS = {
    "course-handbook.md": ["What happens if I submit an assignment late?", "How is the final grade calculated?", "Can I bring notes to the final exam?"],
    "os-lecture-cpu-scheduling.pdf": ["What is the convoy effect?", "How does aging prevent starvation?", "How should the Round Robin time quantum be chosen?"],
  };
  const GENERIC_QUESTIONS = ["Summarise the main points of these documents", "What are the key terms defined?"];

  let docs = [];
  const excluded = new Set();   // document ids the user unticked
  let asking = false;

  const ICON = {
    trash: '<svg viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
    chev: '<svg class="chev" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.1"/><circle cx="12" cy="12" r="9"/></svg>',
    search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>',
    retry: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4"/></svg>',
  };

  // ---------------------------------------------------------------
  // API helpers (works when served by FastAPI or from a static server)
  // ---------------------------------------------------------------
  const API_CANDIDATES = ["", "http://127.0.0.1:8002", "http://localhost:8002"];
  let apiBase = null;
  async function findApi() {
    if (apiBase !== null) return apiBase;
    for (const base of API_CANDIDATES) {
      if (base === "" && location.protocol === "file:") continue;
      try {
        const r = await fetch(base + "/api/health", { signal: AbortSignal.timeout(2500) });
        if ((await r.json())?.status === "ok") return (apiBase = base);
      } catch (_) { /* next */ }
    }
    throw new Error("Couldn't reach the DocLens server. Start it with: uvicorn app.main:app --port 8002");
  }

  async function api(path, opts = {}) {
    const base = await findApi();
    let res;
    try { res = await fetch(base + path, opts); }
    catch (_) { throw new Error("Lost connection to the DocLens server."); }
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch (_) { /* empty */ }
    if (!res.ok) throw new Error(data?.error || `Request failed (${res.status}).`);
    return data;
  }

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg; t.hidden = false;
    clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 2400);
  }

  function md(text) {
    const src = String(text ?? "");
    if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(src, { breaks: true, gfm: true }));
    return src.split(/\n{2,}/).map((p) => `<p>${esc(p)}</p>`).join("");
  }

  // ---------------------------------------------------------------
  // Theme + status
  // ---------------------------------------------------------------
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("dl-theme"); } catch (_) {}
    document.documentElement.dataset.theme = saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("dl-theme", next); } catch (_) {}
    });
  }

  async function checkHealth() {
    const box = $("#status"), text = $("#status-text");
    try {
      const h = await api("/api/health");
      if (!h.api_key_configured) { box.className = "status err"; text.textContent = "API key missing"; return; }
      box.className = "status ok"; text.textContent = `Online · ${h.model}`;
    } catch (e) {
      box.className = "status err"; text.textContent = "Server offline";
    }
  }

  // ---------------------------------------------------------------
  // Library
  // ---------------------------------------------------------------
  async function refreshDocs() {
    try { docs = await api("/api/documents"); }
    catch (e) { docs = []; }
    for (const id of [...excluded]) if (!docs.some((d) => d.id === id)) excluded.delete(id);
    renderDocs();
  }

  function renderDocs() {
    const list = $("#doc-list");
    list.innerHTML = docs.map((d) => `
      <li class="doc ${excluded.has(d.id) ? "off" : ""}" data-id="${d.id}">
        <input type="checkbox" ${excluded.has(d.id) ? "" : "checked"} aria-label="Include ${esc(d.name)} in search" />
        <span class="doc-badge ${d.kind}">${d.kind.toUpperCase()}</span>
        <span class="doc-info"><span class="doc-name" title="${esc(d.name)}">${esc(d.name)}</span>
          <span class="doc-meta">${d.pages} page${d.pages === 1 ? "" : "s"} · ${d.chunks} chunk${d.chunks === 1 ? "" : "s"}</span></span>
        <button class="doc-del" title="Remove">${ICON.trash}</button>
      </li>`).join("");
    $("#lib-empty").hidden = docs.length > 0;
    $("#clear-btn").hidden = docs.length === 0;
    $("#lib-count").textContent = `${docs.length} file${docs.length === 1 ? "" : "s"}`;
    updateScope();
    renderSuggestions();
    if (docs.length) hideFormError();
  }

  function selectedIds() { return docs.filter((d) => !excluded.has(d.id)).map((d) => d.id); }

  function updateScope() {
    const n = selectedIds().length;
    $("#scope-text").textContent = !docs.length ? "No documents yet — add some in the library"
      : n === docs.length ? `Searching all ${docs.length} document${docs.length === 1 ? "" : "s"}`
      : n === 0 ? "No documents selected" : `Searching ${n} of ${docs.length} documents`;
  }

  function renderSuggestions() {
    const box = $("#suggestions");
    if (!docs.length || $("#thread .turn")) { box.hidden = true; return; }
    const qs = docs.flatMap((d) => SAMPLE_QUESTIONS[d.name] || []);
    const list = (qs.length ? qs : GENERIC_QUESTIONS).slice(0, 6);
    box.innerHTML = `<p class="eyebrow">Try asking</p>` + list.map((q) => `<button class="chip" type="button">${esc(q)}</button>`).join("");
    box.hidden = false;
  }

  $("#doc-list").addEventListener("change", (e) => {
    const li = e.target.closest(".doc"); if (!li) return;
    e.target.checked ? excluded.delete(li.dataset.id) : excluded.add(li.dataset.id);
    li.classList.toggle("off", !e.target.checked);
    updateScope();
  });

  $("#doc-list").addEventListener("click", async (e) => {
    const btn = e.target.closest(".doc-del"); if (!btn) return;
    const li = btn.closest(".doc");
    const name = docs.find((d) => d.id === li.dataset.id)?.name;
    try { await api(`/api/documents/${li.dataset.id}`, { method: "DELETE" }); toast(`Removed ${name}`); }
    catch (err) { toast(err.message); }
    refreshDocs();
  });

  $("#clear-btn").addEventListener("click", async () => {
    if (!confirm("Remove all documents from the library?")) return;
    try { await api("/api/documents", { method: "DELETE" }); toast("Library cleared"); }
    catch (err) { toast(err.message); }
    refreshDocs();
  });

  $("#samples-btn").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Indexing samples…';
    try { const added = await api("/api/samples", { method: "POST" }); toast(`Added ${added.length} sample document${added.length === 1 ? "" : "s"}`); }
    catch (err) { toast(err.message); }
    btn.disabled = false;
    btn.innerHTML = '<svg viewBox="0 0 24 24"><path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/></svg> Load sample documents';
    refreshDocs();
  });

  // ---------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------
  function uploadItem(name) {
    const li = document.createElement("li");
    li.className = "upload";
    li.dataset.name = name;
    li.innerHTML = `<div class="u-row"><span class="u-name">${esc(name)}</span><span class="u-state">Indexing…</span></div><div class="u-bar"><i></i></div>`;
    $("#uploads").prepend(li);
    return li;
  }

  function failUpload(li, msg) {
    li.classList.add("err");
    li.innerHTML = `<div class="u-row"><span class="u-name">${esc(li.dataset.name)}</span><span class="u-state">Failed</span><button class="u-close" aria-label="Dismiss">×</button></div><p class="u-msg">${esc(msg)}</p>`;
    $(".u-close", li).addEventListener("click", () => li.remove());
  }

  async function uploadFiles(files) {
    for (const file of files) {
      const li = uploadItem(file.name);
      const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
      if (!ALLOWED.includes(ext)) { failUpload(li, "Unsupported file type. Use PDF, TXT or Markdown."); continue; }
      if (file.size === 0) { failUpload(li, "The file is empty."); continue; }
      if (file.size > MAX_MB * 1024 * 1024) { failUpload(li, `File is larger than ${MAX_MB} MB.`); continue; }
      const form = new FormData();
      form.append("file", file);
      try {
        const doc = await api("/api/documents", { method: "POST", body: form });
        li.remove();
        toast(`Indexed ${doc.name} · ${doc.chunks} chunks`);
        await refreshDocs();
      } catch (err) { failUpload(li, err.message); }
    }
  }

  const dz = $("#dropzone");
  $("#file-input").addEventListener("change", (e) => { uploadFiles([...e.target.files]); e.target.value = ""; });
  dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file-input").click(); } });
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
  // Allow dropping anywhere on the page.
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => { if (!dz.contains(e.target)) { e.preventDefault(); uploadFiles([...e.dataTransfer.files]); } });

  // ---------------------------------------------------------------
  // Asking
  // ---------------------------------------------------------------
  const STOP = new Set("the a an and or of to in on for happen happens is are was were be what which who whom how why when where does do did can could should would will with from that this these those it its as by at about into than then there their my your our me i you we they he she them his her if not no yes".split(" "));

  function highlight(text, question) {
    const terms = [...new Set(question.toLowerCase().match(/[a-z0-9]{3,}/g) || [])].filter((w) => !STOP.has(w));
    let html = esc(text);
    if (!terms.length) return html;
    const re = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
    return html.replace(re, "<mark>$&</mark>");
  }

  function linkCitations(html, n) {
    // Turn [3] into clickable chips (outside of tags only).
    return html.replace(/\[(\d+)\](?![^<]*>)/g, (m, d) => +d >= 1 && +d <= n ? `<button class="cite" data-n="${d}" title="Show source ${d}">${d}</button>` : m);
  }

  // Gemini similarities cluster in ~0.4-0.8: >=0.60 is a strong match, 0.55-0.60 borderline.
  function scoreClass(s) { return s >= 0.6 ? "" : s >= 0.55 ? "mid" : "lo"; }

  function renderSources(sources, question, found, gated) {
    if (!sources.length) return "";
    const label = found ? "Sources" : gated ? "Closest passages (below the relevance threshold)" : "Passages searched";
    return `
      <div class="sources">
        <div class="sources-head"><span class="eyebrow">${label}</span></div>
        <div class="src-list">
          ${sources.map((s) => `
            <div class="src ${s.cited || !found ? "" : "uncited"}" data-n="${s.n}">
              <button class="src-top" type="button" aria-expanded="false">
                <span class="src-n">${s.n}</span>
                <span class="src-title"><b>${esc(s.document_name)}</b> <span>· p.${s.page}</span> ${s.cited ? '<span class="tag-cited">· cited</span>' : ""}</span>
                <span class="score" title="Similarity between your question and this passage"><span class="score-bar"><i class="${scoreClass(s.score)}" style="width:${Math.max(4, Math.round(s.score * 100))}%"></i></span>${s.score.toFixed(2)}</span>
                ${ICON.chev}
              </button>
              <p class="src-text">${highlight(s.text, question)}</p>
            </div>`).join("")}
        </div>
      </div>`;
  }

  function renderAnswer(turn, r) {
    const conf = { high: "High confidence", medium: "Medium confidence", low: "Low confidence", none: "Not in documents" }[r.confidence];
    const time = ((r.retrieval_ms + r.generation_ms) / 1000).toFixed(1);
    const body = r.found
      ? `<div class="answer-body md">${linkCitations(md(r.answer), r.sources.length)}</div>`
      : `<div class="answer-body notfound"><span class="nf-icon">${ICON.search}</span><div class="md">${md(r.answer)}</div></div>`;
    $(".answer", turn).innerHTML = `
      <div class="answer-head">
        <span class="eyebrow">Answer</span>
        <span class="pill ${r.confidence}"><i></i>${conf}</span>
        <span class="pill time" title="Retrieval ${r.retrieval_ms} ms · generation ${r.generation_ms} ms">${time}s</span>
      </div>
      ${body}
      ${renderSources(r.sources, r.question, r.found, r.generation_ms === 0)}`;
  }

  function renderError(turn, msg, question) {
    $(".answer", turn).innerHTML = `
      <div class="error-card"><span class="err-icon">${ICON.alert}</span>
        <div><p>${esc(msg)}</p><button class="btn ghost sm retry">${ICON.retry} Try again</button></div></div>`;
    $(".retry", turn).addEventListener("click", () => { turn.remove(); ask(question); });
  }

  function addTurn(question) {
    $("#intro").hidden = true;
    const turn = document.createElement("section");
    turn.className = "turn";
    turn.innerHTML = `
      <div class="q-bubble"><p>${esc(question)}</p></div>
      <div class="answer"><div class="thinking"><span class="spinner"></span>
        <span class="steps-live"><span class="on">Searching documents</span><span>Writing answer</span></span></div></div>`;
    $("#thread").append(turn);
    turn.scrollIntoView({ behavior: "smooth", block: "start" });
    const steps = $$(".steps-live span", turn);
    turn._timer = setTimeout(() => { steps[0].className = "done"; steps[1].className = "on"; }, 1200);
    return turn;
  }

  function showFormError(msg) { const el = $("#form-error"); el.textContent = msg; el.hidden = false; }
  function hideFormError() { $("#form-error").hidden = true; }

  async function ask(question) {
    hideFormError();
    question = (question ?? $("#question").value).trim();
    if (!question) return showFormError("Type a question first.");
    if (question.length < 3 || !/[a-z]/i.test(question)) return showFormError("Please type a real question.");
    if (!docs.length) return showFormError("Add at least one document to the library first.");
    const ids = selectedIds();
    if (!ids.length) return showFormError("Tick at least one document in the library to search.");
    if (asking) return;

    asking = true;
    $("#ask-btn").disabled = true;
    $("#question").value = ""; autoGrow();
    const turn = addTurn(question);
    try {
      const r = await api("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, document_ids: ids.length === docs.length ? null : ids, top_k: +$("#top-k").value }),
      });
      renderAnswer(turn, r);
    } catch (err) {
      renderError(turn, err.message, question);
    } finally {
      clearTimeout(turn._timer);
      asking = false;
      $("#ask-btn").disabled = false;
    }
  }

  // Source interactions: expand / collapse, citation jump.
  $("#thread").addEventListener("click", (e) => {
    const chip = e.target.closest(".suggestions .chip");
    if (chip) return ask(chip.textContent);
    const top = e.target.closest(".src-top");
    if (top) {
      const src = top.closest(".src");
      src.classList.toggle("open");
      top.setAttribute("aria-expanded", src.classList.contains("open"));
      return;
    }
    const cite = e.target.closest(".cite");
    if (cite) {
      const src = $(`.src[data-n="${cite.dataset.n}"]`, cite.closest(".turn"));
      if (!src) return;
      src.classList.add("open");
      src.scrollIntoView({ behavior: "smooth", block: "center" });
      src.classList.remove("flash"); void src.offsetWidth; src.classList.add("flash");
      setTimeout(() => src.classList.remove("flash"), 1600);
    }
  });

  // Composer
  const qEl = $("#question");
  function autoGrow() { qEl.style.height = "auto"; qEl.style.height = Math.min(qEl.scrollHeight, 180) + "px"; }
  qEl.addEventListener("input", () => { autoGrow(); hideFormError(); });
  qEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); ask(); });

  // ---------------------------------------------------------------
  initTheme();
  checkHealth();
  refreshDocs();
})();
