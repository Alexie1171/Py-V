// chat.js — PY-V chat panel (runs inside the panel's page, Phase 10 + 13)
// Rendering messages, input, scrolling, the live answer while V writes, the
// open-file chip (which file V can read — click x to stop sharing it), Apply to
// file (diff first, the file changes only on Apply), the memory view (facts,
// studied topics), the study bar, "Good answer" (saved for training), and the
// bridge to the extension (panel.ts). Text from V is always put in with
// textContent — never as HTML.

(function () {
  const vscode = acquireVsCodeApi();

  const MODE_LABELS = { chat: "chat", explain: "explain", generate: "write code", debug: "fix", refactor: "improve" };
  const CODE_MODES = ["generate", "debug", "refactor"];
  const START_SERVER = "uvicorn inference.api.main:app --port 8000";

  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendEl = document.getElementById("send");
  const newChatEl = document.getElementById("newChat");
  const memoryEl = document.getElementById("memoryButton");
  const statusEl = document.getElementById("status");
  const fileChipEl = document.getElementById("fileChip");
  const studyBarEl = document.getElementById("studyBar");
  const memoryViewEl = document.getElementById("memoryView");

  // Saved with the page: survives hiding the panel and reloading VS Code.
  // useFile: send the open file with messages (the chip's x turns it off)
  let state = Object.assign({ sessionId: null, messages: [], useFile: true }, vscode.getState() || {});
  let editor = { name: null, where: null };   // the open file, from panel.ts
  let live = null;        // the answer being written: { el, textEl, metaEl, text, mode, load, id, question, target }
  let server = { state: null, detail: null };   // online / starting / offline (server.ts starts her server with the panel)
  const reviews = new Map();   // code block key → its review line (Apply to file)

  function save() {
    vscode.setState(state);
  }

  function newId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  // ─── Rendering ──────────────────────────────────────────────────────────────

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(text, onClick, title) {
    const b = el("button", "link", text);
    if (title) b.title = title;
    b.addEventListener("click", onClick);
    return b;
  }

  function nearBottom() {
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 60;
  }

  function scrollDown(force) {
    if (force || nearBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function renderAll() {
    messagesEl.textContent = "";
    reviews.clear();
    if (state.messages.length === 0) renderWelcome();
    for (const m of state.messages) messagesEl.appendChild(renderMessage(m));
    scrollDown(true);
  }

  function renderWelcome() {
    const box = el("div", "welcome");
    box.appendChild(el("div", "welcome-title", "Hi, I'm V."));
    box.appendChild(el("div", "welcome-text", "Ask me to write, fix, improve or explain code, or just chat."));
    if (server.state === "starting") box.appendChild(el("div", "waking", "Waking up... this takes about a minute."));
    if (server.state === "offline") box.appendChild(offlineHint(server.detail));
    messagesEl.appendChild(box);
  }

  function offlineHint(detail) {
    const box = el("div", "offline-hint");
    box.appendChild(el("div", null, detail || "My server isn't running."));
    box.appendChild(button("Start V", () => vscode.postMessage({ type: "startServer" })));
    box.appendChild(el("div", "info", "Or start it yourself in a terminal in the Py-V folder:"));
    box.appendChild(el("code", "command", START_SERVER));
    return box;
  }

  // A short message that goes away by itself (not part of the chat)
  function notice(text) {
    const n = el("div", "notice", text);
    messagesEl.appendChild(n);
    scrollDown(true);
    setTimeout(() => n.remove(), 4000);
  }

  function renderMessage(m) {
    if (m.role === "user") {
      const box = el("div", "msg user");
      box.appendChild(el("div", "text", m.text));
      return box;
    }
    if (!m.id) m.id = newId();
    const box = el("div", "msg v" + (m.error ? " error" : ""));
    box.dataset.id = m.id;
    box.appendChild(renderMeta(m));
    const body = el("div", "body");
    renderAnswer(body, m);
    box.appendChild(body);
    if (m.sources && m.sources.length) box.appendChild(renderSources(m.sources));
    if (m.stopped) box.appendChild(el("div", "note", "(stopped)"));
    if (m.note) box.appendChild(el("div", "note", m.note));
    if (m.offline) box.appendChild(offlineHint(server.detail));
    if (m.asks && m === state.messages[state.messages.length - 1]) box.appendChild(quickReplies(m.asks));
    if (!m.error && !m.stopped && (m.text || "").trim()) box.appendChild(answerActions(m));
    return box;
  }

  function renderMeta(m) {
    const meta = el("div", "meta");
    if (m.mode) meta.appendChild(el("span", "badge", (MODE_LABELS[m.mode] || m.mode) + (m.language ? ` · ${m.language}` : "")));
    if (m.memories) meta.appendChild(el("span", "info", `memory ${m.memories}`));
    if (m.rag_chunks) meta.appendChild(el("span", "info", `rag ${m.rag_chunks}`));
    if (m.project) meta.appendChild(el("span", "info", `project ${m.project}`));
    if (m.notes) meta.appendChild(el("span", "info", `notes ${m.notes}`));
    if (m.load && m.load !== "free") meta.appendChild(el("span", "info load", `laptop ${m.load}`));
    if (m.file) meta.appendChild(el("span", "info", `read ${m.file}`));
    return meta;
  }

  // Where a web lookup's facts came from (Phase 13) — links open in the browser
  function renderSources(sources) {
    const box = el("div", "sources");
    box.appendChild(el("span", "info", "Sources: "));
    sources.forEach((s, i) => {
      if (!/^https?:\/\//i.test(s.url || "")) return;
      const a = el("a", null, s.title || s.url);
      a.href = s.url;
      a.title = s.url;
      box.appendChild(a);
      if (i < sources.length - 1) box.appendChild(document.createTextNode(" · "));
    });
    return box;
  }

  // "Want me to look that up online?" — typed yes / no works too; these just save typing
  function quickReplies(asks) {
    const box = el("div", "quick");
    for (const text of asks) {
      box.appendChild(button(text, () => {
        inputEl.value = text;
        send();
      }));
    }
    return box;
  }

  // "Good answer": saved as a training example for V's next training run (only what you approve)
  function answerActions(m) {
    const box = el("div", "answer-actions");
    if (m.approvedId) {
      box.appendChild(el("span", "info", "Saved for training"));
      box.appendChild(button("undo", () => vscode.postMessage({ type: "unapprove", key: m.id, approvedId: m.approvedId })));
    } else {
      box.appendChild(button("Good answer", () => {
        vscode.postMessage({ type: "approve", key: m.id, question: m.question || "", answer: m.text, mode: m.mode, language: m.language || null });
      }, "Save this question and answer as a training example for my next training run. Only answers you approve are used."));
    }
    return box;
  }

  function findMessage(id) {
    return state.messages.find((m) => m.id === id);
  }

  function rerender(m) {
    const node = [...messagesEl.querySelectorAll(".msg.v")].find((n) => n.dataset.id === m.id);
    if (node) node.replaceWith(renderMessage(m));
  }

  // ─── The open file ─────────────────────────────────────────────────────────

  // Which file V can read with the next message: its name (+ the selected
  // lines). V reads it only when the message is about it (a selection always);
  // x stops sending it, clicking the chip again turns it back on.
  function renderFileChip() {
    fileChipEl.textContent = "";
    fileChipEl.hidden = !editor.name;
    if (!editor.name) return;
    fileChipEl.classList.toggle("off", !state.useFile);
    const label = el("span", "file-name", editor.name + (editor.where ? ` · ${editor.where}` : ""));
    fileChipEl.appendChild(label);
    const toggle = el("button", "link file-toggle", state.useFile ? "×" : "share");
    toggle.title = state.useFile
      ? "V reads this file when your message is about it (a selection always). Click to stop sharing it."
      : "Not shared with V. Click to share it again.";
    toggle.addEventListener("click", () => {
      state.useFile = !state.useFile;
      save();
      renderFileChip();
    });
    fileChipEl.title = state.useFile ? "Shared with V" : "Not shared with V";
    fileChipEl.appendChild(toggle);
  }

  // An answer: code blocks (with Copy / Insert / Apply to file) and text. Fenced
  // ``` blocks are code; in the code modes V often answers with bare code, so
  // code-looking paragraphs become code blocks there too.
  function renderAnswer(container, m) {
    let n = 0;
    for (const block of splitBlocks(m.text || "", m.mode)) {
      container.appendChild(block.code !== undefined ? codeBlock(block.code, `${m.id}:${n++}`, m.target) : textBlock(block.text));
    }
  }

  function splitBlocks(text, mode) {
    if (text.includes("```")) {
      const blocks = [];
      const parts = text.split("```");
      parts.forEach((part, i) => {
        if (i % 2 === 1) {
          const lines = part.split("\n");
          if (lines.length > 1 && /^[\w+#.-]*$/.test(lines[0].trim())) lines.shift();   // language tag
          const code = lines.join("\n").replace(/\s+$/, "");
          if (code) blocks.push({ code });
        } else if (part.trim()) {
          blocks.push({ text: part.trim() });
        }
      });
      return blocks;
    }
    if (!CODE_MODES.includes(mode)) return [{ text }];

    const blocks = [];
    for (const para of text.split(/\n\s*\n/)) {
      if (!para.trim()) continue;
      const isCode = para.split("\n").some(looksLikeCode);
      const last = blocks[blocks.length - 1];
      if (isCode && last && last.code !== undefined) last.code += "\n\n" + para.replace(/\s+$/, "");
      else if (isCode) blocks.push({ code: para.replace(/\s+$/, "") });
      else blocks.push({ text: para.trim() });
    }
    return blocks;
  }

  function looksLikeCode(line) {
    return /^\s*(def |class |import |from \S+ import|return\b|if |elif |else:|for |while |try:|except|finally:|with |@|#|print\()/.test(line)
      || /^\s*[\w.\[\]]+\s*[+\-*/]?=[^=]/.test(line)
      || /[:([{,;]\s*$/.test(line)
      || /^\s{2,}\S/.test(line);
  }

  function codeBlock(code, key, target) {
    const box = el("div", "code");
    const actions = el("div", "actions");
    actions.appendChild(button("Copy", () => vscode.postMessage({ type: "copy", code })));
    actions.appendChild(button("Insert at cursor", () => vscode.postMessage({ type: "insert", code })));
    actions.appendChild(button("Apply to file", () => {
      setReview(key, { working: true });
      vscode.postMessage({ type: "apply", key, code, target: target || null });
    }, target ? `Show the change to ${target.name} first; it changes only when you click Apply` :
      "Show the change to the open file first; it changes only when you click Apply"));
    const pre = el("pre");
    pre.appendChild(el("code", null, code));
    const review = el("div", "review");
    review.hidden = true;
    reviews.set(key, review);
    box.appendChild(actions);
    box.appendChild(pre);
    box.appendChild(review);
    return box;
  }

  // The review line under a code block: working → the change (Apply / Discard) → the result
  function setReview(key, r) {
    const review = reviews.get(key);
    if (!review) return;
    review.textContent = "";
    review.hidden = false;
    if (r.working) {
      review.appendChild(el("span", "info", "Working out the change..."));
    } else if (r.summary) {
      review.appendChild(el("span", null, `The diff is open: ${r.summary}. `));
      review.appendChild(button("Apply", () => {
        setReview(key, { working: true });
        vscode.postMessage({ type: "applyChange", key });
      }, "Change the file (not saved yet - Ctrl+S saves, Ctrl+Z undoes)"));
      review.appendChild(document.createTextNode(" "));
      review.appendChild(button("Discard", () => vscode.postMessage({ type: "discardChange", key })));
    } else {
      review.appendChild(el("span", r.ok ? "info ok" : "info", r.text || ""));
    }
  }

  // Plain text with `inline code` and **bold**; line breaks kept (CSS pre-wrap)
  function textBlock(text) {
    const p = el("div", "text");
    for (const piece of text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/)) {
      if (!piece) continue;
      if (piece.startsWith("`") && piece.endsWith("`") && piece.length > 2) p.appendChild(el("code", null, piece.slice(1, -1)));
      else if (piece.startsWith("**") && piece.endsWith("**") && piece.length > 4) p.appendChild(el("strong", null, piece.slice(2, -2)));
      else p.appendChild(document.createTextNode(piece));
    }
    return p;
  }

  // ─── The answer being written ──────────────────────────────────────────────

  function startLive(id, question) {
    const box = el("div", "msg v live");
    const metaEl = el("div", "meta");
    const textEl = el("div", "text", "");
    textEl.appendChild(el("span", "thinking", "V is thinking..."));
    box.appendChild(metaEl);
    box.appendChild(textEl);
    const welcome = messagesEl.querySelector(".welcome");
    if (welcome) welcome.remove();
    messagesEl.appendChild(box);
    live = { el: box, textEl, metaEl, text: "", mode: null, load: null, id, question, target: null };
    scrollDown(true);
  }

  function finishLive(message) {
    if (!live) return;
    const stick = nearBottom();
    message.id = live.id;
    message.question = live.question;
    message.target = live.target;
    state.messages.push(message);      // first: quick replies show only under the newest answer
    live.el.replaceWith(renderMessage(message));
    live = null;
    save();
    setBusy(false);
    if (stick) scrollDown(true);
  }

  function setBusy(busy) {
    sendEl.textContent = busy ? "Stop" : "Send";
    sendEl.classList.toggle("stop", busy);
  }

  // ─── Sending ────────────────────────────────────────────────────────────────

  // /memory, /stop-server, /start-server, /stop-study, /help — handled by the
  // extension, never sent to the brain, not saved with the chat
  function command(text) {
    inputEl.value = "";
    autoGrow();
    const welcome = messagesEl.querySelector(".welcome");
    if (welcome) welcome.remove();
    messagesEl.appendChild(renderMessage({ role: "user", text }));
    scrollDown(true);
    vscode.postMessage({ type: "command", name: text.slice(1).trim().toLowerCase() });
  }

  function send() {
    if (live) return;
    const text = inputEl.value.trim();
    if (!text) return;
    if (text.startsWith("/")) {
      command(text);
      return;
    }
    if (server.state !== "online") {
      notice(server.state === "starting" ? "I'm still waking up, give me a moment." : "My server isn't running yet.");
      return;
    }
    inputEl.value = "";
    autoGrow();
    const last = state.messages[state.messages.length - 1];
    if (last && last.asks) {
      delete last.asks;
      rerender(last);
    }
    state.messages.push({ role: "user", text });
    save();
    const welcome = messagesEl.querySelector(".welcome");
    if (welcome) welcome.remove();
    messagesEl.appendChild(renderMessage({ role: "user", text }));
    const id = newId();
    startLive(id, text);
    setBusy(true);
    vscode.postMessage({ type: "send", text, useFile: state.useFile, id });
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + "px";
  }

  sendEl.addEventListener("click", () => {
    if (live) vscode.postMessage({ type: "stop" });
    else send();
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", autoGrow);

  newChatEl.addEventListener("click", () => vscode.postMessage({ type: "newChat" }));
  memoryEl.addEventListener("click", () => {
    if (!memoryViewEl.hidden) closeMemory();
    else vscode.postMessage({ type: "memory" });
  });

  // ─── Memory view (10.4) + what she studied (Phase 13) ──────────────────────

  function closeMemory() {
    memoryViewEl.hidden = true;
    memoryViewEl.textContent = "";
  }

  function renderMemory(msg) {
    memoryViewEl.textContent = "";
    memoryViewEl.hidden = false;
    const head = el("div", "memory-head");
    head.appendChild(el("span", "memory-title", "What I remember"));
    head.appendChild(button("Close", closeMemory));
    memoryViewEl.appendChild(head);

    if (!msg.enabled) {
      memoryViewEl.appendChild(el("div", "info", "My memory is switched off (config memory.enabled)."));
    } else {
      const facts = msg.facts || [];
      memoryViewEl.appendChild(el("div", "memory-section", `Facts (${facts.length})`));
      if (!facts.length) memoryViewEl.appendChild(el("div", "info", "Nothing yet. Tell me things like \"remember that ...\"."));
      for (const f of facts) {
        const row = el("div", "memory-row");
        row.dataset.fact = String(f.id);
        row.appendChild(el("span", "memory-text", f.text));
        row.appendChild(el("span", "info", new Date(f.created * 1000).toLocaleDateString()));
        row.appendChild(button("Forget", () => {
          row.classList.add("going");
          vscode.postMessage({ type: "forget", id: f.id });
        }, "Forget this for good"));
        memoryViewEl.appendChild(row);
      }
      memoryViewEl.appendChild(el("div", "info memory-foot",
        `${msg.messages || 0} messages saved across ${msg.sessions || 0} chats (used for facts and earlier code; they stay on this computer).`));
    }

    const learning = msg.learning;
    if (learning) {
      const topics = learning.topics || [];
      memoryViewEl.appendChild(el("div", "memory-section", `Studied topics (${topics.length})`));
      if (!topics.length) memoryViewEl.appendChild(el("div", "info", "None yet. Try \"learn about asyncio for 30 minutes\"."));
      for (const t of topics) {
        const row = el("div", "memory-row topic");
        row.dataset.topic = String(t.id);
        const text = el("div", "memory-text");
        text.appendChild(el("strong", null, t.topic));
        text.appendChild(document.createTextNode(` · ${Math.round(t.minutes)} min · ${t.notes} notes`));
        if (t.covered && t.covered.length) text.appendChild(el("div", "info", "Covered: " + t.covered.slice(0, 6).join(", ")));
        if (t.next && t.next.length) text.appendChild(el("div", "info", "Next: " + t.next.slice(0, 4).join(", ")));
        row.appendChild(text);
        row.appendChild(button(t.approved ? "Used for training · stop" : "Use for training", () =>
          vscode.postMessage({ type: "approveTopic", id: t.id, approved: !t.approved }),
          t.approved ? "Her notes on this topic go into the next training run. Click to stop that."
                     : "Let her notes on this topic go into the next training run (a Kaggle run you start)."));
        row.appendChild(button("Forget", () => {
          row.classList.add("going");
          vscode.postMessage({ type: "forgetTopic", id: t.id });
        }, "Forget this topic and its notes"));
        memoryViewEl.appendChild(row);
      }
      memoryViewEl.appendChild(el("div", "info memory-foot",
        `Answers you marked "Good answer": ${learning.approved || 0} (training examples for the next training run).`));
    }

    const project = msg.project;
    if (project) {
      memoryViewEl.appendChild(el("div", "memory-section", "Project search"));
      const state = { ready: "ready", indexing: "reading your files...", paused: "paused while I'm busy", idle: "waiting", error: "failed" }[project.status] || project.status;
      memoryViewEl.appendChild(el("div", "info",
        `${project.files || 0} files, ${project.pieces || 0} pieces (${project.embedded || 0} with meaning search) - ${state}. ` +
        "It stays on this computer; /index reads it again."));
    }
  }

  // ─── Study bar (Phase 13) ──────────────────────────────────────────────────

  function renderStudy(s) {
    studyBarEl.textContent = "";
    studyBarEl.hidden = !(s && s.running);
    if (studyBarEl.hidden) return;
    const left = Math.max(0, Math.round(s.minutes_left));
    studyBarEl.appendChild(el("span", null,
      `Studying ${s.topic} · ${left} min left · ${s.notes} note${s.notes === 1 ? "" : "s"}` + (s.paused ? " · paused (laptop busy)" : "")));
    studyBarEl.appendChild(button("Stop", () => vscode.postMessage({ type: "stopStudy" }), "Stop studying; what she learned so far is kept"));
  }

  // ─── Messages from the extension ───────────────────────────────────────────

  window.addEventListener("message", (event) => {
    const msg = event.data || {};
    switch (msg.type) {
      case "session":
        state.sessionId = msg.sessionId;
        save();
        break;

      case "status": {
        const changed = server.state !== msg.state || server.detail !== (msg.detail || null);
        server = { state: msg.state, detail: msg.detail || null };
        statusEl.className = "status " + server.state;
        statusEl.title = { online: "V is ready", starting: "V is waking up", offline: "V's server isn't running" }[server.state] || "";
        if (changed && state.messages.length === 0 && !live) renderAll();
        if (server.state !== "online") renderStudy(null);
        break;
      }

      case "editor":
        editor = { name: msg.name || null, where: msg.where || null };
        renderFileChip();
        break;

      case "target":
        if (live && live.id === msg.id) live.target = msg.target;
        break;

      case "start":
        if (!live) return;
        live.mode = msg.mode;
        live.load = msg.load;
        live.language = msg.language;
        live.file = msg.file;
        live.metaEl.replaceWith((live.metaEl = renderMeta({ mode: msg.mode, load: msg.load, language: msg.language, file: msg.file })));
        break;

      case "lookupStatus":   // she is looking it up online (Phase 13)
        if (!live || live.text) return;
        live.textEl.textContent = "";
        live.textEl.appendChild(el("span", "thinking", msg.text + " ..."));
        break;

      case "piece":
        if (!live) return;
        live.text += msg.text;
        live.textEl.textContent = live.text;
        scrollDown(false);
        break;

      case "done":
        finishLive({
          role: "v", text: msg.response, mode: msg.mode, memories: msg.memories, rag_chunks: msg.rag_chunks,
          project: msg.project, notes: msg.notes, load: msg.load, note: msg.note, language: msg.language,
          file: msg.file, sources: msg.sources || null, asks: msg.asks || null,
        });
        if (msg.study) renderStudy(msg.study);
        break;

      case "stopped":
        if (live) finishLive({ role: "v", text: live.text.trim(), mode: live.mode, load: live.load, language: live.language, file: live.file, stopped: true });
        break;

      case "error":
        if (live) {
          finishLive({
            role: "v", error: true, offline: !!msg.offline,
            text: msg.offline ? "I can't reach my server." : `Something went wrong: ${msg.detail}`,
          });
        }
        break;

      case "proposal":
        setReview(msg.key, { summary: msg.summary });
        break;

      case "proposalDone":
        setReview(msg.key, { ok: !!msg.ok, text: msg.text });
        break;

      case "approved": {
        const m = findMessage(msg.key);
        if (m) {
          m.approvedId = msg.approvedId;
          save();
          rerender(m);
        }
        break;
      }

      case "unapproved": {
        const m = findMessage(msg.key);
        if (m) {
          delete m.approvedId;
          save();
          rerender(m);
        }
        break;
      }

      case "memoryList":
        renderMemory(msg);
        break;

      case "forgotten": {
        const row = memoryViewEl.querySelector(`[data-fact="${Number(msg.id)}"]`);
        if (row) row.remove();
        break;
      }

      case "topicForgotten": {
        const row = memoryViewEl.querySelector(`[data-topic="${Number(msg.id)}"]`);
        if (row) row.remove();
        break;
      }

      case "memoryError":
        memoryViewEl.querySelectorAll(".going").forEach((row) => row.classList.remove("going"));
        notice(msg.detail);
        break;

      case "study":
        renderStudy(msg);
        break;

      case "reply": {
        const box = el("div", "msg v reply");
        box.appendChild(el("div", "text", msg.text));
        messagesEl.appendChild(box);
        scrollDown(true);
        break;
      }

      case "cleared":
        if (live) {
          live.el.remove();
          live = null;
          setBusy(false);
        }
        state.messages = [];
        save();
        renderAll();
        break;
    }
  });

  renderAll();
  renderFileChip();
  vscode.postMessage({ type: "ready", sessionId: state.sessionId });
})();
