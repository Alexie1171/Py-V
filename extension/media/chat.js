// chat.js — PY-V chat panel (runs inside the panel's page, Phase 10)
// Rendering messages, input, scrolling, the live answer while V writes, and
// the bridge to the extension (panel.ts). Text from V is always put in with
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
  const statusEl = document.getElementById("status");

  // Saved with the page: survives hiding the panel and reloading VS Code
  let state = vscode.getState() || { sessionId: null, messages: [] };
  let live = null;        // the answer being written: { el, textEl, metaEl, text, mode, load }
  let server = { state: null, detail: null };   // online / starting / offline (server.ts starts her server with the panel)

  function save() {
    vscode.setState(state);
  }

  // ─── Rendering ──────────────────────────────────────────────────────────────

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function nearBottom() {
    return messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 60;
  }

  function scrollDown(force) {
    if (force || nearBottom()) messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function renderAll() {
    messagesEl.textContent = "";
    if (state.messages.length === 0) renderWelcome();
    for (const m of state.messages) messagesEl.appendChild(renderMessage(m));
    scrollDown(true);
  }

  function renderWelcome() {
    const box = el("div", "welcome");
    box.appendChild(el("div", "welcome-title", "Hi, I'm V."));
    box.appendChild(el("div", "welcome-text", "Ask me to write, fix, improve or explain Python code, or just chat."));
    if (server.state === "starting") box.appendChild(el("div", "waking", "Waking up... this takes about a minute."));
    if (server.state === "offline") box.appendChild(offlineHint(server.detail));
    messagesEl.appendChild(box);
  }

  function offlineHint(detail) {
    const box = el("div", "offline-hint");
    box.appendChild(el("div", null, detail || "My server isn't running."));
    const start = el("button", "link", "Start V");
    start.addEventListener("click", () => vscode.postMessage({ type: "startServer" }));
    box.appendChild(start);
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
    const box = el("div", "msg v" + (m.error ? " error" : ""));
    box.appendChild(renderMeta(m));
    const body = el("div", "body");
    renderAnswer(body, m.text, m.mode);
    box.appendChild(body);
    if (m.stopped) box.appendChild(el("div", "note", "(stopped)"));
    if (m.note) box.appendChild(el("div", "note", m.note));
    if (m.offline) box.appendChild(offlineHint(server.detail));
    return box;
  }

  function renderMeta(m) {
    const meta = el("div", "meta");
    if (m.mode) meta.appendChild(el("span", "badge", MODE_LABELS[m.mode] || m.mode));
    if (m.memories) meta.appendChild(el("span", "info", `memory ${m.memories}`));
    if (m.rag_chunks !== undefined && m.rag_chunks !== null) meta.appendChild(el("span", "info", `rag ${m.rag_chunks}`));
    if (m.load && m.load !== "free") meta.appendChild(el("span", "info load", `laptop ${m.load}`));
    return meta;
  }

  // An answer: code blocks (with Copy / Insert) and text. Fenced ``` blocks are
  // code; in the code modes V often answers with bare code, so code-looking
  // paragraphs become code blocks there too.
  function renderAnswer(container, text, mode) {
    for (const block of splitBlocks(text || "", mode)) {
      container.appendChild(block.code !== undefined ? codeBlock(block.code) : textBlock(block.text));
    }
  }

  function splitBlocks(text, mode) {
    if (text.includes("```")) {
      const blocks = [];
      const parts = text.split("```");
      parts.forEach((part, i) => {
        if (i % 2 === 1) {
          const lines = part.split("\n");
          if (lines.length > 1 && /^[\w+-]*$/.test(lines[0].trim())) lines.shift();   // language tag
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
      || /[:([{,]\s*$/.test(line)
      || /^\s{2,}\S/.test(line);
  }

  function codeBlock(code) {
    const box = el("div", "code");
    const actions = el("div", "actions");
    const copy = el("button", "link", "Copy");
    copy.addEventListener("click", () => vscode.postMessage({ type: "copy", code }));
    const insert = el("button", "link", "Insert at cursor");
    insert.addEventListener("click", () => vscode.postMessage({ type: "insert", code }));
    actions.appendChild(copy);
    actions.appendChild(insert);
    const pre = el("pre");
    pre.appendChild(el("code", null, code));
    box.appendChild(actions);
    box.appendChild(pre);
    return box;
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

  function startLive() {
    const box = el("div", "msg v live");
    const metaEl = el("div", "meta");
    const textEl = el("div", "text", "");
    textEl.appendChild(el("span", "thinking", "V is thinking..."));
    box.appendChild(metaEl);
    box.appendChild(textEl);
    const welcome = messagesEl.querySelector(".welcome");
    if (welcome) welcome.remove();
    messagesEl.appendChild(box);
    live = { el: box, textEl, metaEl, text: "", mode: null, load: null };
    scrollDown(true);
  }

  function finishLive(message) {
    if (!live) return;
    const stick = nearBottom();
    const node = renderMessage(message);
    live.el.replaceWith(node);
    live = null;
    state.messages.push(message);
    save();
    setBusy(false);
    if (stick) scrollDown(true);
  }

  function setBusy(busy) {
    sendEl.textContent = busy ? "Stop" : "Send";
    sendEl.classList.toggle("stop", busy);
  }

  // ─── Sending ────────────────────────────────────────────────────────────────

  function send() {
    if (live) return;
    const text = inputEl.value.trim();
    if (!text) return;
    if (server.state !== "online") {
      notice(server.state === "starting" ? "I'm still waking up, give me a moment." : "My server isn't running yet.");
      return;
    }
    inputEl.value = "";
    autoGrow();
    state.messages.push({ role: "user", text });
    save();
    const welcome = messagesEl.querySelector(".welcome");
    if (welcome) welcome.remove();
    messagesEl.appendChild(renderMessage({ role: "user", text }));
    startLive();
    setBusy(true);
    vscode.postMessage({ type: "send", text });
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
        break;
      }

      case "start":
        if (!live) return;
        live.mode = msg.mode;
        live.load = msg.load;
        live.metaEl.replaceWith((live.metaEl = renderMeta({ mode: msg.mode, load: msg.load })));
        break;

      case "piece":
        if (!live) return;
        live.text += msg.text;
        live.textEl.textContent = live.text;
        scrollDown(false);
        break;

      case "done":
        finishLive({
          role: "v", text: msg.response, mode: msg.mode, memories: msg.memories,
          rag_chunks: msg.rag_chunks, load: msg.load, note: msg.note,
        });
        break;

      case "stopped":
        if (live) finishLive({ role: "v", text: live.text.trim(), mode: live.mode, load: live.load, stopped: true });
        break;

      case "error":
        if (live) {
          finishLive({
            role: "v", error: true, offline: !!msg.offline,
            text: msg.offline ? "I can't reach my server." : `Something went wrong: ${msg.detail}`,
          });
        }
        break;

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
  vscode.postMessage({ type: "ready", sessionId: state.sessionId });
})();
