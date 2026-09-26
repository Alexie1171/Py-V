/**
 * panel.ts — PY-V VS Code Extension (Phase 10)
 * The chat panel in the sidebar: creates the page (chat_view.ts), passes
 * messages between the page (media/chat.js) and the server (api.ts), does the
 * editor work the page asks for (insert code, copy, apply a change to a file —
 * file_update.ts), follows the open file (file_context.ts — sent with each
 * message unless the user turned it off), sends the workspace folder for
 * project search (10.5 — indexed when V is ready and after saves), shows what
 * V remembers and has studied (memory view), follows a study session (Phase
 * 13), and tells the server manager (server.ts) when the panel opens and
 * closes — her server follows it.
 *
 * Page → extension: ready {sessionId}, send {text, useFile, id}, stop, newChat, insert {code}, copy {code},
 *                   apply {key, code, target} / applyChange {key} / discardChange {key} (10.3),
 *                   memory / forget {id} (10.4), approve {key, question, answer, mode, language} /
 *                   unapprove {key, approvedId} / approveTopic {id, approved} / forgetTopic {id} / stopStudy (13),
 *                   startServer, command {name} (/memory, /stop-study, /index, /stop-server, /start-server, /help)
 * Extension → page: session {sessionId}, status {state, detail}, editor {name, where} | {name: null},
 *                   target {id, target} (the file an answer read), start / lookupStatus {text} / piece / done /
 *                   error (the answer), stopped, cleared, proposal {key, summary} / proposalDone {key, ok, text},
 *                   memoryList {enabled, facts, messages, sessions, learning, project} / forgotten {id} /
 *                   topicForgotten {id} / memoryError {detail}, approved {key, approvedId} / unapproved {key},
 *                   study {running, topic, minutes_left, notes, paused},
 *                   reply {text} (V's answer to a command — not saved, not sent to the brain)
 */

import * as vscode from "vscode";
import {
  approveAnswer,
  approveTopic,
  chatStream,
  forgetFact,
  forgetTopic,
  getLearning,
  getMemory,
  indexProject,
  projectStatus,
  stopStudy,
  studyStatus,
  unapproveAnswer,
} from "./api";
import { getChatHtml } from "./chat_view";
import { describeEditor, isReadable, readOpenFile } from "./file_context";
import { ChangeReviewer, Target } from "./file_update";
import { insertCode } from "./provider";
import { ServerManager, ServerState } from "./server";

export class ChatViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
  public static readonly viewId = "pyv.chatView";

  private view?: vscode.WebviewView;
  private sessionId = newSessionId();
  private stopAnswer?: () => void;
  private status: { state: ServerState; detail?: string } = { state: "offline" };
  private editor?: vscode.TextEditor;          // the last file the user was in (typing in the panel takes the focus away)
  private editorTimer?: ReturnType<typeof setTimeout>;
  private readonly reviewer = new ChangeReviewer();
  private studyTimer?: ReturnType<typeof setInterval>;
  private indexTimer?: ReturnType<typeof setTimeout>;

  constructor(private readonly extensionUri: vscode.Uri, private readonly server: ServerManager) {
    server.onState((state, detail) => {
      const cameOnline = state === "online" && this.status.state !== "online";
      this.status = { state, detail };
      this.post({ type: "status", state, detail });
      if (cameOnline) {
        this.indexProjectSoon(0);   // project search: bring the index up to date (in the background)
        this.watchStudy();          // a study session may already be running
      }
    });
  }

  dispose(): void {
    this.reviewer.dispose();
    clearInterval(this.studyTimer);
    clearTimeout(this.indexTimer);
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
      // Project search: saved files go into the index (a while after the last save)
      vscode.workspace.onDidSaveTextDocument(() => this.indexProjectSoon(20000)),
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

  /** The workspace folder for project search (10.5): the open file's folder, else the first one; trusted only. */
  private projectRoot(): string | undefined {
    if (!vscode.workspace.getConfiguration("pyv").get<boolean>("projectSearch", true) || !vscode.workspace.isTrusted) {
      return undefined;
    }
    const editor = this.currentEditor();
    const folder =
      (editor && vscode.workspace.getWorkspaceFolder(editor.document.uri)) ?? vscode.workspace.workspaceFolders?.[0];
    return folder?.uri.scheme === "file" ? folder.uri.fsPath : undefined;
  }

  private indexProjectSoon(delayMs: number): void {
    clearTimeout(this.indexTimer);
    this.indexTimer = setTimeout(async () => {
      const root = this.projectRoot();
      if (root && this.status.state === "online") {
        try {
          await indexProject(root);
        } catch {
          // the server went away — tried again when it is back
        }
      }
    }, delayMs);
  }

  /** While a study session runs, the page's study bar is kept up to date. */
  private watchStudy(): void {
    if (this.studyTimer) {
      return;
    }
    const tick = async () => {
      try {
        const s = await studyStatus();
        this.post({ type: "study", ...s });
        if (!s.running) {
          clearInterval(this.studyTimer);
          this.studyTimer = undefined;
        }
      } catch {
        clearInterval(this.studyTimer);
        this.studyTimer = undefined;
        this.post({ type: "study", running: false });
      }
    };
    this.studyTimer = setInterval(tick, 10000);
    void tick();
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
        this.send(String(msg.text ?? ""), msg.useFile !== false, String(msg.id ?? ""));
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
      case "apply":
        await this.proposeChange(String(msg.key ?? ""), String(msg.code ?? ""), msg.target as Target | undefined);
        break;
      case "applyChange": {
        const key = String(msg.key ?? "");
        try {
          this.post({ type: "proposalDone", key, ...(await this.reviewer.apply(key)) });
        } catch (err) {
          this.post({ type: "proposalDone", key, ok: false, text: `Couldn't apply it: ${errorText(err)}` });
        }
        break;
      }
      case "discardChange":
        await this.reviewer.discard(String(msg.key ?? ""));
        this.post({ type: "proposalDone", key: String(msg.key ?? ""), ok: false, text: "Discarded - the file is unchanged." });
        break;
      case "memory":
        await this.showMemory();
        break;
      case "forget":
        try {
          await forgetFact(Number(msg.id));
          this.post({ type: "forgotten", id: Number(msg.id) });
        } catch (err) {
          this.post({ type: "memoryError", detail: errorText(err) });
        }
        break;
      case "approve":
        try {
          const saved = await approveAnswer(String(msg.question ?? ""), String(msg.answer ?? ""), msg.mode ?? null,
                                            msg.language ?? null, this.sessionId);
          this.post({ type: "approved", key: msg.key, approvedId: saved.id });
        } catch (err) {
          this.post({ type: "memoryError", detail: `Couldn't save it: ${errorText(err)}` });
        }
        break;
      case "unapprove":
        try {
          await unapproveAnswer(String(msg.approvedId ?? ""));
        } catch {
          // already gone
        }
        this.post({ type: "unapproved", key: msg.key });
        break;
      case "approveTopic":
        try {
          await approveTopic(Number(msg.id), !!msg.approved);
          await this.showMemory();
        } catch (err) {
          this.post({ type: "memoryError", detail: errorText(err) });
        }
        break;
      case "forgetTopic":
        try {
          await forgetTopic(Number(msg.id));
          this.post({ type: "topicForgotten", id: Number(msg.id) });
        } catch (err) {
          this.post({ type: "memoryError", detail: errorText(err) });
        }
        break;
      case "stopStudy":
        await this.stopStudying();
        break;
    }
  }

  private send(text: string, useFile: boolean, id: string): void {
    if (!text.trim()) {
      return;
    }
    this.stop();
    // The open file as it is right now; the server takes what the message needs
    const editor = useFile ? this.currentEditor() : undefined;
    const file = editor ? readOpenFile(editor) : undefined;
    const project = this.projectRoot();
    if (editor && file) {
      // Saved with the answer: "Apply to file" changes this file, and the selection it read
      const target: Target = {
        uri: editor.document.uri.toString(),
        name: file.name,
        selection: file.selection,
        selectionLine: file.selection_line,
      };
      this.post({ type: "target", id, target });
    }
    this.stopAnswer = chatStream(this.sessionId, text, { file, project }, (event) => {
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
        if (event.result.study?.running) {
          this.watchStudy();
        }
        return;
      }
      if (event.kind === "status") {
        this.post({ type: "lookupStatus", text: event.text });
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

  /** "Apply to file" on a code block: work out the change and show it as a diff (nothing written yet). */
  private async proposeChange(key: string, code: string, target: Target | undefined): Promise<void> {
    if (this.status.state !== "online") {
      this.post({ type: "proposalDone", key, ok: false, text: "My server isn't running - start it, then try again." });
      return;
    }
    try {
      const summary = await this.reviewer.show(key, code, target, this.currentEditor());
      this.post({ type: "proposal", key, summary });
    } catch (err) {
      this.post({ type: "proposalDone", key, ok: false, text: errorText(err) });
    }
  }

  /** The memory view: facts (each with Forget), studied topics (training approval, Forget), project search. */
  private async showMemory(): Promise<void> {
    if (this.status.state !== "online") {
      this.post({ type: "memoryError", detail: "My server isn't running - start it to see what I remember." });
      return;
    }
    const root = this.projectRoot();
    const [memory, learning, project] = await Promise.allSettled([
      getMemory(),
      getLearning(),
      root ? projectStatus(root) : Promise.resolve(undefined),
    ]);
    if (memory.status === "rejected") {
      this.post({ type: "memoryError", detail: errorText(memory.reason) });
      return;
    }
    this.post({
      type: "memoryList",
      ...memory.value,
      learning: learning.status === "fulfilled" ? learning.value : null,
      project: project.status === "fulfilled" ? project.value ?? null : null,
    });
  }

  private async stopStudying(): Promise<void> {
    try {
      const { reply } = await stopStudy();
      this.post({ type: "reply", text: reply });
    } catch (err) {
      this.post({ type: "reply", text: `Couldn't stop it: ${errorText(err)}` });
    }
    this.watchStudy();
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
      case "memory":
        await this.showMemory();
        break;
      case "stop-study":
        await this.stopStudying();
        break;
      case "index": {
        const root = this.projectRoot();
        if (!root) {
          this.post({ type: "reply", text: "Project search is off (setting pyv.projectSearch) or there's no trusted folder open." });
          break;
        }
        try {
          const s = await indexProject(root);
          this.post({ type: "reply", text: `Indexing ${root} in the background (${s.files ?? 0} files so far). I'll use it when you ask about your project.` });
        } catch (err) {
          this.post({ type: "reply", text: `Couldn't index it: ${errorText(err)}` });
        }
        break;
      }
      default:
        this.post({ type: "reply", text: COMMANDS_HELP });
    }
  }

  private async insert(code: string): Promise<void> {
    const editor = this.currentEditor();
    if (!editor) {
      vscode.window.showWarningMessage("V: open a file first, then insert the code.");
      return;
    }
    await insertCode(editor, code);
  }
}

const COMMANDS_HELP =
  "Commands: /memory shows what I remember and have studied, /stop-study stops a study session, " +
  "/index re-reads your project for project search, /stop-server stops my server (the panel stays open), " +
  "/start-server starts it again, /help shows this. To study: \"learn about asyncio for 30 minutes\".";

function newSessionId(): string {
  return `panel-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
