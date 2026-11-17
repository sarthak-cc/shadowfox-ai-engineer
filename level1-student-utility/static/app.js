/* StudyPilot frontend — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  // ---------------------------------------------------------------
  // Tool metadata
  // ---------------------------------------------------------------
  const TOOLS = {
    summarize: {
      eyebrow: "01 — Summarise",
      title: "Turn messy notes into <em>revision gold</em>",
      lede: "Paste lecture notes, a textbook passage or your own scribbles. You'll get a TL;DR, key points, definitions and a revision tip.",
      endpoint: "/api/summarize",
      button: "Summarise",
      loading: ["Reading your notes…", "Pulling out the key ideas…", "Defining key terms…", "Polishing the summary…"],
    },
    quiz: {
      eyebrow: "02 — Quiz",
      title: "Test yourself <em>before</em> the exam does",
      lede: "Generate multiple-choice questions straight from your notes, answer them here and see instant explanations.",
      endpoint: "/api/quiz",
      button: "Build quiz",
      loading: ["Reading your notes…", "Writing questions…", "Crafting tricky distractors…", "Checking answers…"],
    },
    improve: {
      eyebrow: "03 — Improve",
      title: "Make a weak answer <em>exam-ready</em>",
      lede: "Paste the question and your answer. You'll get an honest score, what's missing, and a rewritten model answer that keeps your ideas.",
      endpoint: "/api/improve",
      button: "Improve answer",
      loading: ["Reading the question…", "Marking your answer…", "Finding the gaps…", "Writing a model answer…"],
    },
    explain: {
      eyebrow: "04 — Explain",
      title: "Finally <em>get</em> that concept",
      lede: "Name any topic and choose your level. You'll get intuition first, then precision, an everyday analogy, a worked example and self-check questions.",
      endpoint: "/api/explain",
      button: "Explain it",
      loading: ["Thinking about the best angle…", "Building intuition…", "Finding an analogy…", "Working an example…"],
    },
  };

  const SAMPLES = {
    summarize: {
      "sum-notes": `Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into chemical energy stored in glucose. It takes place mainly in the chloroplasts of leaf mesophyll cells. The overall equation is 6CO2 + 6H2O + light -> C6H12O6 + 6O2.

There are two stages. The light-dependent reactions happen in the thylakoid membranes: chlorophyll absorbs light (mostly red and blue wavelengths), water is split (photolysis) releasing O2, and the energy is used to make ATP and NADPH. The light-independent reactions (Calvin cycle) happen in the stroma: the enzyme RuBisCO fixes CO2 onto RuBP, and ATP and NADPH from stage one are used to reduce the products into G3P, some of which becomes glucose.

Limiting factors include light intensity, CO2 concentration and temperature. At low light, increasing light increases the rate until another factor becomes limiting. Temperature affects enzyme activity, so too high a temperature denatures enzymes like RuBisCO and the rate falls. Plants use glucose for respiration, to make starch for storage, cellulose for cell walls, and proteins when combined with nitrates from the soil.`,
    },
    quiz: {
      "quiz-notes": `The French Revolution (1789-1799) was a period of radical political and social change in France. Causes included a severe financial crisis caused by debt from wars (including support for the American Revolution), an unfair tax system where the clergy (First Estate) and nobility (Second Estate) paid little tax while the Third Estate carried the burden, poor harvests that raised bread prices, and Enlightenment ideas from thinkers like Rousseau and Voltaire.

In May 1789 King Louis XVI called the Estates-General. The Third Estate broke away and formed the National Assembly, taking the Tennis Court Oath not to disband until France had a constitution. On 14 July 1789 crowds stormed the Bastille, a royal prison seen as a symbol of tyranny. In August the Assembly issued the Declaration of the Rights of Man and of the Citizen, proclaiming liberty and equality.

The monarchy was abolished in 1792 and Louis XVI was executed in 1793. The radical Jacobins under Robespierre led the Reign of Terror (1793-94), during which thousands were executed by guillotine. Robespierre was himself executed in 1794. The Directory then governed until 1799, when Napoleon Bonaparte seized power in a coup.`,
    },
    improve: {
      "imp-q": "Explain why the sky appears blue during the day.",
      "imp-a": "The sky is blue because it reflects the ocean. Also sunlight has lots of colours and blue is one of them so we see blue. At sunset it is orange because the sun is lower.",
    },
    explain: {
      "exp-topic": "Recursion",
      "exp-subject": "Computer Science",
    },
  };

  // Client-side validation mirrors the server rules for instant feedback.
  const RULES = {
    summarize: [{ id: "sum-notes", label: "Notes", minWords: 30 }],
    quiz: [{ id: "quiz-notes", label: "Notes", minWords: 40 }],
    improve: [
      { id: "imp-q", label: "Question", minWords: 3 },
      { id: "imp-a", label: "Answer", minWords: 5 },
    ],
    explain: [{ id: "exp-topic", label: "Topic", minWords: 1 }],
  };

  // ---------------------------------------------------------------
  // State
  // ---------------------------------------------------------------
  let current = "summarize";
  let lastResult = null;   // { tool, data }
  let inFlight = null;     // AbortController
  let loadingTimer = null;

  // ---------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function md(text) {
    const src = String(text ?? "");
    if (window.marked && window.DOMPurify) {
      return DOMPurify.sanitize(marked.parse(src, { breaks: true, gfm: true }));
    }
    return src.split(/\n{2,}/).map((p) => `<p>${esc(p).replace(/\n/g, "<br>")}</p>`).join("");
  }

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => (t.hidden = true), 2200);
  }

  const wordCount = (s) => (s.trim() ? s.trim().split(/\s+/).length : 0);

  const ICON = {
    check: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
    minus: '<svg viewBox="0 0 24 24"><path d="M5 12h14"/></svg>',
    alert: '<svg viewBox="0 0 24 24"><path d="M12 8v5M12 16.5v.1"/><circle cx="12" cy="12" r="9"/></svg>',
    bulb: '<svg viewBox="0 0 24 24"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.4 1 1.1 1 1.8V16h5v-.3c0-.7.4-1.4 1-1.8A6 6 0 0 0 12 3z"/></svg>',
    flask: '<svg viewBox="0 0 24 24"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-9V3"/></svg>',
    retry: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 2.3-5.7M4 4v4h4"/></svg>',
  };

  // ---------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------
  function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("sp-theme"); } catch (_) {}
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = saved || (prefersDark ? "dark" : "light");
    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("sp-theme", next); } catch (_) {}
    });
  }

  // ---------------------------------------------------------------
  // Tool switching
  // ---------------------------------------------------------------
  function selectTool(tool) {
    if (!TOOLS[tool]) return;
    current = tool;
    const meta = TOOLS[tool];
    $$(".tool").forEach((b) => {
      const on = b.dataset.tool === tool;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on);
    });
    $$(".fields").forEach((f) => (f.hidden = f.dataset.for !== tool));
    $("#tool-eyebrow").textContent = meta.eyebrow;
    $("#tool-title").innerHTML = meta.title;
    $("#tool-lede").textContent = meta.lede;
    $("#submit-btn .btn-label").textContent = meta.button;
    hideFormError();
    if (lastResult?.tool !== tool) showEmpty();
    else render(lastResult.tool, lastResult.data);
    if (location.hash !== "#" + tool) history.replaceState(null, "", "#" + tool);
  }

  // ---------------------------------------------------------------
  // Form helpers
  // ---------------------------------------------------------------
  function segmentedValue(name) {
    const seg = $(`.fields[data-for="${current}"] .segmented[data-name="${name}"]`);
    return $("button.on", seg)?.dataset.value;
  }

  function collectPayload() {
    switch (current) {
      case "summarize":
        return { notes: $("#sum-notes").value, length: segmentedValue("length") };
      case "quiz":
        return { notes: $("#quiz-notes").value, num_questions: +$("#quiz-n").value, difficulty: segmentedValue("difficulty") };
      case "improve":
        return { question: $("#imp-q").value, answer: $("#imp-a").value };
      case "explain":
        return { topic: $("#exp-topic").value, subject: $("#exp-subject").value.trim() || null, level: segmentedValue("level") };
    }
  }

  function validate() {
    $$(".invalid").forEach((el) => el.classList.remove("invalid"));
    for (const rule of RULES[current]) {
      const el = document.getElementById(rule.id);
      const value = el.value.trim();
      let msg = null;
      if (!value) msg = `${rule.label} cannot be empty.`;
      else if (wordCount(value) < rule.minWords) msg = `${rule.label} is too short — please provide at least ${rule.minWords} words (you have ${wordCount(value)}).`;
      if (msg) {
        el.classList.add("invalid");
        el.focus();
        return msg;
      }
    }
    return null;
  }

  function showFormError(msg) {
    const el = $("#form-error");
    el.innerHTML = ICON.alert + `<span>${esc(msg)}</span>`;
    el.hidden = false;
    el.style.animation = "none"; el.offsetHeight; el.style.animation = "";
  }
  function hideFormError() { $("#form-error").hidden = true; }

  function updateCounter(input) {
    const c = $(`[data-count="${input.id}"]`);
    if (!c) return;
    const max = +input.getAttribute("maxlength");
    c.textContent = `${input.value.length.toLocaleString()} / ${max.toLocaleString()}`;
    c.classList.toggle("warn", input.value.length > max * 0.92);
  }

  function updateRange() {
    const r = $("#quiz-n");
    $("#nq-out").textContent = r.value;
    r.style.setProperty("--fill", ((r.value - r.min) / (r.max - r.min)) * 100 + "%");
  }

  // ---------------------------------------------------------------
  // Output states
  // ---------------------------------------------------------------
  function showEmpty() {
    $("#output-toolbar").hidden = true;
    $("#output").innerHTML = `
      <div class="empty">
        <div class="empty-art" aria-hidden="true"><span></span><span></span><span></span></div>
        <h2>Your results land here</h2>
        <p>Pick a tool, add your material and hit <b>${esc(TOOLS[current].button)}</b>. Not sure where to start? Use <b>Try a sample</b>.</p>
      </div>`;
  }

  function showLoading() {
    $("#output-toolbar").hidden = true;
    $("#output-panel").setAttribute("aria-busy", "true");
    const steps = TOOLS[current].loading;
    $("#output").innerHTML = `
      <div class="loading">
        <div class="loading-status"><span class="spinner"></span><span id="loading-text">${esc(steps[0])}</span></div>
        <div class="skel h"></div>
        <div class="skel block"></div>
        <div class="skel" style="width:92%"></div>
        <div class="skel" style="width:85%"></div>
        <div class="skel" style="width:70%"></div>
        <div class="skel block"></div>
        <div class="skel" style="width:80%"></div>
      </div>`;
    let i = 0;
    clearInterval(loadingTimer);
    loadingTimer = setInterval(() => {
      i = Math.min(i + 1, steps.length - 1);
      const el = $("#loading-text");
      if (el) el.textContent = steps[i];
    }, 2200);
  }

  function showError(msg) {
    $("#output-toolbar").hidden = true;
    $("#output").innerHTML = `
      <div class="error-card" role="alert">
        <div class="err-icon">${ICON.alert}</div>
        <div>
          <h3>Something went wrong</h3>
          <p>${esc(msg)}</p>
          <button class="btn ghost" id="retry-btn">${ICON.retry} Try again</button>
        </div>
      </div>`;
    $("#retry-btn").addEventListener("click", () => $("#tool-form").requestSubmit());
  }

  // ---------------------------------------------------------------
  // Submit
  // ---------------------------------------------------------------
  async function submit(e) {
    e.preventDefault();
    hideFormError();
    const err = validate();
    if (err) return showFormError(err);

    inFlight?.abort();
    inFlight = new AbortController();
    const tool = current;
    const btn = $("#submit-btn");
    btn.disabled = true;
    $(".btn-label", btn).innerHTML = `<span class="spinner"></span>&nbsp;Working…`;
    showLoading();

    try {
      const res = await fetch((await findApi()) + TOOLS[tool].endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
        signal: inFlight.signal,
      });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
      if (!res.ok) {
        const msg = data?.error || `Request failed (${res.status}). Please try again.`;
        if (res.status === 422) { showFormError(msg); if (current === tool) showEmpty(); }
        else if (current === tool) showError(msg);
        return;
      }
      lastResult = { tool, data };
      if (current === tool) render(tool, data);
    } catch (ex) {
      if (ex.name === "AbortError") return;
      if (current === tool) showError("Couldn't reach the StudyPilot server. Start it with: uvicorn app.main:app --port 8001");
    } finally {
      clearInterval(loadingTimer);
      $("#output-panel").setAttribute("aria-busy", "false");
      btn.disabled = false;
      $(".btn-label", btn).textContent = TOOLS[current].button;
    }
  }

  // ---------------------------------------------------------------
  // Renderers
  // ---------------------------------------------------------------
  function render(tool, data) {
    $("#output-toolbar").hidden = false;
    $("#output-label").textContent = { summarize: "Summary", quiz: "Quiz", improve: "Feedback", explain: "Explanation" }[tool];
    const html = { summarize: renderSummary, quiz: renderQuiz, improve: renderImprove, explain: renderExplain }[tool](data);
    $("#output").innerHTML = `<article class="result">${html}</article>`;
    if (tool === "quiz") wireQuiz(data);
    if (tool === "improve") requestAnimationFrame(() => $$(".ring").forEach((r) => r.style.setProperty("--v", r.dataset.v)));
    $("#output-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderSummary(d) {
    return `
      <h2 class="r-title">${esc(d.title)}</h2>
      <div class="callout"><b class="k">TL;DR</b>${esc(d.tldr)}</div>
      <h3>Key points</h3>
      <ol class="points">${d.key_points.map((p) => `<li><span>${esc(p)}</span></li>`).join("")}</ol>
      <h3>Key terms</h3>
      <div class="terms">${d.key_terms.map((t) => `<div class="term"><b>${esc(t.term)}</b><p>${esc(t.definition)}</p></div>`).join("")}</div>
      <h3>Revision tip</h3>
      <div class="callout yellow">${esc(d.revision_tip)}</div>`;
  }

  function renderQuiz(d) {
    const L = "ABCD";
    return `
      <h2 class="r-title">${esc(d.title)}</h2>
      <div class="quiz-progress"><div class="bar"><i id="qbar"></i></div><span id="qcount">0 / ${d.questions.length} answered</span></div>
      ${d.questions.map((q, i) => `
        <div class="q" data-q="${i}" style="animation-delay:${i * 60}ms">
          <div class="q-num">QUESTION ${String(i + 1).padStart(2, "0")}</div>
          <div class="q-text">${esc(q.question)}</div>
          <div class="opts">${q.options.map((o, j) => `<button class="opt" data-o="${j}"><span class="letter">${L[j]}</span><span>${esc(o)}</span></button>`).join("")}</div>
        </div>`).join("")}
      <div id="quiz-done"></div>`;
  }

  function wireQuiz(d) {
    let answered = 0, correct = 0;
    const total = d.questions.length;
    $$(".q").forEach((qEl) => {
      const q = d.questions[+qEl.dataset.q];
      $$(".opt", qEl).forEach((btn) => btn.addEventListener("click", () => {
        const pick = +btn.dataset.o;
        $$(".opt", qEl).forEach((b, j) => {
          b.disabled = true;
          if (j === q.correct_index) b.classList.add("correct");
          else if (j === pick) b.classList.add("wrong");
          else b.classList.add("dim");
        });
        const right = pick === q.correct_index;
        answered++; if (right) correct++;
        qEl.insertAdjacentHTML("beforeend", `<div class="explain"><b>${right ? "Correct!" : "Not quite."}</b> ${esc(q.explanation)}</div>`);
        $("#qbar").style.width = (answered / total) * 100 + "%";
        $("#qcount").textContent = `${answered} / ${total} answered · ${correct} correct`;
        if (answered === total) {
          const pct = Math.round((correct / total) * 100);
          const msg = pct === 100 ? "Perfect score — you've nailed this topic." : pct >= 70 ? "Strong work. Review the ones you missed." : pct >= 40 ? "Getting there. Re-read the explanations and try again." : "Worth another pass through your notes.";
          $("#quiz-done").innerHTML = `<div class="quiz-done"><div class="big">${correct}/${total}</div><p>${msg}</p><button class="btn ghost" id="retake">${ICON.retry} Retake quiz</button></div>`;
          $("#retake").addEventListener("click", () => render("quiz", d));
        }
      }));
    });
  }

  function renderImprove(d) {
    const delta = d.score_after - d.score_before;
    const ring = (v, label, color) => `
      <div class="score"><div class="ring" data-v="${v * 10}" style="--c:${color}"><span>${v}</span></div>
      <div><small>${label}</small><b>${v}/10</b></div></div>`;
    return `
      <div class="scores">
        ${ring(d.score_before, "Your answer", "var(--bad)")}
        <svg class="arrow" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        ${ring(d.score_after, "Improved", "var(--good)")}
        ${delta > 0 ? `<span class="delta">+${delta} points</span>` : ""}
      </div>
      <h3>Feedback</h3>
      <div class="cards-2">
        <div class="card good"><div class="k">${ICON.plus} Strengths</div><ul>${d.feedback.strengths.map((s) => `<li>${esc(s)}</li>`).join("") || "<li>—</li>"}</ul></div>
        <div class="card bad"><div class="k">${ICON.minus} Gaps</div><ul>${d.feedback.gaps.map((s) => `<li>${esc(s)}</li>`).join("") || "<li>—</li>"}</ul></div>
      </div>
      <h3>Rewritten answer</h3>
      <div class="model-answer md">${md(d.improved_answer)}</div>
      <h3>What changed</h3>
      <ul class="changes">${d.changes.map((c) => `<li>${ICON.check}<span>${esc(c)}</span></li>`).join("")}</ul>`;
  }

  function renderExplain(d) {
    return `
      <h2 class="r-title">${esc(d.title)}</h2>
      <div class="callout"><b class="k">In one sentence</b>${esc(d.one_liner)}</div>
      <h3>Explanation</h3>
      <div class="md">${md(d.explanation)}</div>
      <h3>Think of it like…</h3>
      <div class="callout yellow">${md(d.analogy)}</div>
      <h3>Example</h3>
      <div class="card md">${md(d.example)}</div>
      <h3>Common mistakes</h3>
      <div class="card bad"><ul>${d.common_mistakes.map((m) => `<li>${esc(m)}</li>`).join("")}</ul></div>
      <h3>Check yourself</h3>
      ${d.check_questions.map((q) => `<div class="check">${esc(q)}</div>`).join("")}`;
  }

  // ---------------------------------------------------------------
  // Markdown export
  // ---------------------------------------------------------------
  function toMarkdown({ tool, data: d }) {
    switch (tool) {
      case "summarize":
        return `# ${d.title}\n\n> ${d.tldr}\n\n## Key points\n${d.key_points.map((p) => `- ${p}`).join("\n")}\n\n## Key terms\n${d.key_terms.map((t) => `- **${t.term}** — ${t.definition}`).join("\n")}\n\n## Revision tip\n${d.revision_tip}\n`;
      case "quiz":
        return `# ${d.title}\n\n` + d.questions.map((q, i) => `### ${i + 1}. ${q.question}\n${q.options.map((o, j) => `${"ABCD"[j]}) ${o}`).join("\n")}\n\n**Answer:** ${"ABCD"[q.correct_index]} — ${q.explanation}\n`).join("\n");
      case "improve":
        return `# Answer review\n\nScore: ${d.score_before}/10 → ${d.score_after}/10\n\n## Strengths\n${d.feedback.strengths.map((s) => `- ${s}`).join("\n")}\n\n## Gaps\n${d.feedback.gaps.map((s) => `- ${s}`).join("\n")}\n\n## Improved answer\n${d.improved_answer}\n\n## Changes\n${d.changes.map((s) => `- ${s}`).join("\n")}\n`;
      case "explain":
        return `# ${d.title}\n\n> ${d.one_liner}\n\n${d.explanation}\n\n## Analogy\n${d.analogy}\n\n## Example\n${d.example}\n\n## Common mistakes\n${d.common_mistakes.map((s) => `- ${s}`).join("\n")}\n\n## Check yourself\n${d.check_questions.map((s) => `- ${s}`).join("\n")}\n`;
    }
  }

  async function copyResult() {
    if (!lastResult) return;
    try { await navigator.clipboard.writeText(toMarkdown(lastResult)); toast("Copied as Markdown"); }
    catch (_) { toast("Couldn't access the clipboard"); }
  }

  function downloadResult() {
    if (!lastResult) return;
    const blob = new Blob([toMarkdown(lastResult)], { type: "text/markdown" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `studypilot-${lastResult.tool}.md` });
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ---------------------------------------------------------------
  // Health check
  // ---------------------------------------------------------------
  // The UI is normally served by FastAPI (same origin). If it is opened another way
  // (e.g. VS Code Live Server or a file:// URL), fall back to the default backend address.
  const API_CANDIDATES = ["", "http://127.0.0.1:8001", "http://localhost:8001"];
  let apiBase = null;

  async function findApi() {
    if (apiBase !== null) return apiBase;
    for (const base of API_CANDIDATES) {
      if (base === "" && location.protocol === "file:") continue;
      try {
        const r = await fetch(base + "/api/health", { signal: AbortSignal.timeout(2500) });
        const h = await r.json();
        if (h?.status === "ok") { apiBase = base; return base; }
      } catch (_) { /* try next */ }
    }
    throw new Error("backend not found");
  }

  async function checkHealth() {
    const box = $("#status"), text = $("#status-text");
    try {
      const r = await fetch((await findApi()) + "/api/health");
      const h = await r.json();
      if (!h.api_key_configured) { box.className = "status err"; text.textContent = "API key missing"; return; }
      box.className = "status ok"; text.textContent = `Online · ${h.model}`;
    } catch (_) {
      box.className = "status err"; text.textContent = "Server offline";
    }
  }

  // ---------------------------------------------------------------
  // Wire up
  // ---------------------------------------------------------------
  function init() {
    initTheme();
    $$(".tool").forEach((b) => b.addEventListener("click", () => selectTool(b.dataset.tool)));
    $$(".segmented").forEach((seg) => seg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", seg).forEach((x) => x.classList.toggle("on", x === b));
    }));
    $$("textarea, input[type=text]").forEach((el) => {
      el.addEventListener("input", () => { updateCounter(el); el.classList.remove("invalid"); });
      updateCounter(el);
    });
    $("#quiz-n").addEventListener("input", updateRange); updateRange();
    $("#tool-form").addEventListener("submit", submit);
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); $("#tool-form").requestSubmit(); }
    });
    $("#sample-btn").addEventListener("click", () => {
      for (const [id, val] of Object.entries(SAMPLES[current])) {
        const el = document.getElementById(id); el.value = val; updateCounter(el); el.classList.remove("invalid");
      }
      hideFormError();
      toast("Sample loaded — hit " + TOOLS[current].button);
    });
    $("#copy-btn").addEventListener("click", copyResult);
    $("#download-btn").addEventListener("click", downloadResult);

    const fromHash = location.hash.slice(1);
    selectTool(TOOLS[fromHash] ? fromHash : "summarize");
    checkHealth();
  }

  init();
})();
