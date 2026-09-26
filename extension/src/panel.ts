/**
 * panel.ts — PY-V VS Code Extension (Phase 10)
 * The chat panel in the sidebar: creates the page (chat_view.ts), passes
 * messages between the page (media/chat.js) and the server (api.ts), does the
 * editor work the page asks for (insert code, copy), follows the open file
 * (file_context.ts — sent with each message unless the user turned it off),
 * and tells the server manager (server.ts) when the panel opens and closes —
 * her server follows it.
 *
 * Page → extension: ready {sessionId}, send {text, useFile}, stop, newChat, insert {code}, copy {code}, startServer,
 *                   command {name} (/stop-server, /start-server typed in the chat)
 * Extension → page: session {sessionId}, status {state, detail}, editor {name, where} | {name: null},
 *                   start / piece / done / error (the answer), stopped, cleared,
 *                   reply {text} (V's answer to a command — not saved, not sent to the brain)
 */

import * as vscode from "vscode";
import { chatStream } from "./api";
import { getChatHtml } from "./chat_view";
import { describeEditor, isReadable, readOpenFile } from "./file_context";
import { insertCode } from "./provider";
import { ServerManager, ServerState } from "./server";

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = "pyv.chatView";

  private view?: vscode.WebviewView;
  private sessionId = newSessionId();
  private stopAnswer?: () => void;
  private status: { state: ServerState; detail?: string } = { state: "offline" };
  private editor?: vscode.TextEditor;          // the last file the user was in (typing in the panel takes the focus away)
  private editorTimer?: ReturnType<typeof setTimeout>;

  constructor(private readonly extensionUri: vscode.Uri, private readonly server: ServerManager) {
    server.onState((state, detail) => {
      this.status = { state, detail };
      this.post({ type: "status", state, detail });
    });
  }

  /** Follow the open file for the panel's chip; call once (extension.ts). */
  trackEditor(): vscode.Disposable {
    this.editor = isReadable(vscode.window.activeTextEditor) ? vscode.window.activeTextEditor : undefined;
    return vscode.Disposable.from(
      vscode.window.onDidChangeActiveTextEditor((editor) => {
        if (isReadable(editor)) {
          this.editor = editor;
        }
        this.postEditor();
      }),
      vscode.window.onDidChangeTextEditorSelection((e) => {
        if (isReadable(e.textEditor)) {
          this.editor = e.textEditor;
          this.postEditor();
        }
      }),
      // A closed tab: its editor is no longer visible (the document itself can stay loaded a while)
      vscode.window.onDidChangeVisibleTextEditors(() => this.postEditor()),
      { dispose: () => clearTimeout(this.editorTimer) }
    );
  }

  /** The file the user works in: the last one they were in while it's still on screen, else any on screen. */
  private currentEditor(): vscode.TextEditor | undefined {
    const visible = vscode.window.visibleTextEditors.filter(isReadable);
    const last = this.editor;
    if (last && visible.some((e) => e.document === last.document)) {
      return visible.find((e) => e === last) ?? visible.find((e) => e.document === last.document);
    }
    const active = vscode.window.activeTextEditor;
    this.editor = isReadable(active) ? active : visible[0];
    return this.editor;
  }

  /** The chip above the input: which file V can read (a selection changes often — wait a moment). */
  private postEditor(): void {
    clearTimeout(this.editorTimer);
    this.editorTimer = setTimeout(() => {
      const editor = this.currentEditor();
      this.post(editor ? { type: "editor", ...describeEditor(editor) } : { type: "editor", name: null });
    }, 150);
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
        this.postEditor();
        break;
      case "startServer":
        await this.server.startByUser();
        break;
      case "command":
        await this.command(String(msg.name ?? ""));
        break;
      case "send":
        this.send(String(msg.text ?? ""), msg.useFile !== false);
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

  private send(text: string, useFile: boolean): void {
    if (!text.trim()) {
      return;
    }
    this.stop();
    // The open file as it is right now; the server takes what the message needs
    const editor = useFile ? this.currentEditor() : undefined;
    const file = editor ? readOpenFile(editor) : undefined;
    this.stopAnswer = chatStream(this.sessionId, text, file, (event) => {
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

  /** Chat commands — handled here, never sent to the brain. */
  private async command(name: string): Promise<void> {
    switch (name) {
      case "stop-server":
        this.stop();
        this.post({ type: "reply", text: await this.server.stopByUser() });
        break;
      case "start-server":
        this.post({ type: "reply", text: await this.server.startByUser() });
        break;
      default:
        this.post({ type: "reply", text: COMMANDS_HELP });
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

const COMMANDS_HELP = "Commands: /stop-server stops my server (the panel stays open), /start-server starts it again, /help shows this.";

function newSessionId(): string {
  return `panel-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
