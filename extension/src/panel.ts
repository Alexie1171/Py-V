/**
 * panel.ts — PY-V VS Code Extension (Phase 10)
 * The chat panel in the sidebar: creates the page (chat_view.ts), passes
 * messages between the page (media/chat.js) and the server (api.ts), does the
 * editor work the page asks for (insert code, copy), and tells the server
 * manager (server.ts) when the panel opens and closes — her server follows it.
 *
 * Page → extension: ready {sessionId}, send {text}, stop, newChat, insert {code}, copy {code}, startServer
 * Extension → page: session {sessionId}, status {state, detail}, start / piece / done / error (the answer), stopped, cleared
 */

import * as vscode from "vscode";
import { chatStream } from "./api";
import { getChatHtml } from "./chat_view";
import { insertCode } from "./provider";
import { ServerManager, ServerState } from "./server";

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = "pyv.chatView";

  private view?: vscode.WebviewView;
  private sessionId = newSessionId();
  private stopAnswer?: () => void;
  private status: { state: ServerState; detail?: string } = { state: "offline" };

  constructor(private readonly extensionUri: vscode.Uri, private readonly server: ServerManager) {
    server.onState((state, detail) => {
      this.status = { state, detail };
      this.post({ type: "status", state, detail });
    });
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")],
    };
    view.webview.html = getChatHtml(view.webview, this.extensionUri);
    view.webview.onDidReceiveMessage((msg) => this.onMessage(msg));

    // Her server follows the panel: open → start, closed a while → stop
    if (view.visible) {
      void this.server.panelOpened();
    }
    view.onDidChangeVisibility(() => {
      if (view.visible) {
        void this.server.panelOpened();
      } else {
        this.server.panelClosed();
      }
    });
    view.onDidDispose(() => {
      this.stop();
      this.server.panelClosed();
      this.view = undefined;
    });
  }

  private post(message: object): void {
    this.view?.webview.postMessage(message);
  }

  private async onMessage(msg: any): Promise<void> {
    switch (msg?.type) {
      case "ready":
        // The page keeps its session across reloads — use it if it has one
        if (typeof msg.sessionId === "string" && msg.sessionId) {
          this.sessionId = msg.sessionId;
        }
        this.post({ type: "session", sessionId: this.sessionId });
        this.post({ type: "status", ...this.status });
        break;
      case "startServer":
        this.server.start();
        break;
      case "send":
        this.send(String(msg.text ?? ""));
        break;
      case "stop":
        this.stop();
        break;
      case "newChat":
        this.stop();
        this.sessionId = newSessionId();
        this.post({ type: "session", sessionId: this.sessionId });
        this.post({ type: "cleared" });
        break;
      case "insert":
        await this.insert(String(msg.code ?? ""));
        break;
      case "copy":
        await vscode.env.clipboard.writeText(String(msg.code ?? ""));
        vscode.window.setStatusBarMessage("V: code copied", 2000);
        break;
    }
  }

  private send(text: string): void {
    if (!text.trim()) {
      return;
    }
    this.stop();
    this.stopAnswer = chatStream(this.sessionId, text, (event) => {
      if (event.kind === "done" || event.kind === "error") {
        this.stopAnswer = undefined;
      }
      if (event.kind === "error") {
        const offline = /ECONNREFUSED|ECONNRESET|ENOTFOUND|socket hang up/i.test(event.detail);
        this.post({ type: "error", detail: event.detail, offline });
        return;
      }
      if (event.kind === "done") {
        this.post({ type: "done", ...event.result });
        return;
      }
      this.post({ type: event.kind, ...event });
    });
  }

  private stop(): void {
    if (this.stopAnswer) {
      this.stopAnswer();
      this.stopAnswer = undefined;
      this.post({ type: "stopped" });
    }
  }

  private async insert(code: string): Promise<void> {
    const editor = vscode.window.activeTextEditor ?? vscode.window.visibleTextEditors[0];
    if (!editor) {
      vscode.window.showWarningMessage("V: open a file first, then insert the code.");
      return;
    }
    await insertCode(editor, code);
  }
}

function newSessionId(): string {
  return `panel-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
