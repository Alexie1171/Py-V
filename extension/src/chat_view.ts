/**
 * chat_view.ts — PY-V VS Code Extension (Phase 10)
 * The chat panel's page: HTML skeleton, security policy and links to
 * media/chat.css (look) and media/chat.js (everything that runs inside the
 * panel — rendering, input, scrolling). No styles or scripts inline.
 */

import * as vscode from "vscode";

export function getChatHtml(webview: vscode.Webview, extensionUri: vscode.Uri): string {
  const media = (file: string) => webview.asWebviewUri(vscode.Uri.joinPath(extensionUri, "media", file));
  const nonce = makeNonce();

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link href="${media("chat.css")}" rel="stylesheet">
  <title>V</title>
</head>
<body>
  <header class="bar">
    <span id="status" class="status" title="Server status"></span>
    <span class="name">V</span>
    <button id="newChat" class="link" title="Start a new chat (V still remembers facts from old chats)">New chat</button>
  </header>
  <main id="messages" class="messages" aria-live="polite"></main>
  <div id="fileChip" class="file-chip" hidden></div>
  <footer class="composer">
    <textarea id="input" rows="1" placeholder="Message V (Enter to send, Shift+Enter for a new line)"></textarea>
    <button id="send" class="send">Send</button>
  </footer>
  <script nonce="${nonce}" src="${media("chat.js")}"></script>
</body>
</html>`;
}

function makeNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}
