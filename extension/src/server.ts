/**
 * server.ts — PY-V VS Code Extension (Phase 10)
 * V's server follows the chat panel (owner, 2026-09-26): opening the panel
 * starts it (`python -m uvicorn inference.api.main:app` in the Py-V folder,
 * ~1 minute to load the brain), and once the panel has been closed for
 * `pyv.stopServerAfterSeconds` (default 2 minutes — quick trips to Explorer
 * don't reload the brain) it stops, freeing RAM and the GPU. Closing VS Code
 * stops it too. Only a server this extension started is ever stopped — one
 * that was already running (e.g. from a terminal) is used and left alone.
 * Server output: Output → "V Server".
 */

import * as vscode from "vscode";
import { ChildProcess, execFileSync, spawn } from "child_process";
import * as fs from "fs";
import * as path from "path";
import { checkHealth } from "./api";

export type ServerState = "online" | "starting" | "offline";

type Listener = (state: ServerState, detail?: string) => void;

const POLL_MS = 2000;

export class ServerManager implements vscode.Disposable {
  private proc?: ChildProcess;
  private stopTimer?: NodeJS.Timeout;
  private pollTimer?: NodeJS.Timeout;
  private state: ServerState = "offline";
  private detail?: string;
  private heldByUser = false; // after /stop-server: no automatic start until /start-server
  private readonly listeners = new Set<Listener>();
  private readonly output = vscode.window.createOutputChannel("V Server");

  onState(listener: Listener): vscode.Disposable {
    this.listeners.add(listener);
    listener(this.state, this.detail);
    return new vscode.Disposable(() => this.listeners.delete(listener));
  }

  /** The panel became visible: cancel a pending stop, watch the server, start it if needed. */
  async panelOpened(): Promise<void> {
    this.cancelStop();
    this.startPolling();
    if (this.proc) {
      return; // ours — starting or running
    }
    if (await isUp()) {
      this.set("online");
      return;
    }
    if (settings().manageServer && !this.heldByUser) {
      this.start();
    } else {
      this.set("offline", this.heldByUser ? "My server is stopped. Type /start-server to start it." : undefined);
    }
  }

  /** /stop-server in the chat: stop now and stay stopped until /start-server. Returns V's reply. */
  async stopByUser(): Promise<string> {
    this.heldByUser = true;
    if (this.proc) {
      this.stop("/stop-server");
      this.set("offline", "My server is stopped. Type /start-server to start it.");
      return "Okay, my server is stopped. Type /start-server when you want me back.";
    }
    if (await isUp()) {
      return "That server was started outside VS Code (in a terminal?), so I can't stop it from here. Press Ctrl+C in its terminal.";
    }
    this.set("offline", "My server is stopped. Type /start-server to start it.");
    return "My server isn't running.";
  }

  /** /start-server in the chat (or the Start V button). Returns V's reply. */
  async startByUser(): Promise<string> {
    this.heldByUser = false;
    if (this.proc) {
      return this.state === "online" ? "I'm already running." : "I'm already starting, give me a moment.";
    }
    if (await isUp()) {
      this.set("online");
      return "I'm already running.";
    }
    this.start();
    return this.proc ? "Starting my server... this takes about a minute." : this.detail || "I couldn't start my server.";
  }

  /** The panel was closed or hidden: stop our server after the grace time. */
  panelClosed(): void {
    this.stopPolling();
    if (!this.proc) {
      return;
    }
    this.cancelStop();
    const seconds = Math.max(0, settings().stopAfterSeconds);
    this.output.appendLine(`Panel closed — stopping V's server in ${seconds} s unless it opens again.`);
    this.stopTimer = setTimeout(() => this.stop("panel closed"), seconds * 1000);
  }

  /** Start V's server (the panel's "Start V" button, or panelOpened). */
  start(): void {
    if (this.proc) {
      return;
    }
    const project = findProject();
    if (!project.folder) {
      this.set("offline", project.problem);
      return;
    }
    const { python, host, port } = settings();
    const args = ["-m", "uvicorn", "inference.api.main:app", "--host", host, "--port", port];
    this.output.appendLine(`Starting V's server: ${python} ${args.join(" ")}  (in ${project.folder})`);

    const proc = spawn(python, args, {
      cwd: project.folder,
      windowsHide: true,
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
    });
    this.proc = proc;
    this.set("starting");

    proc.stdout?.on("data", (d: Buffer) => this.output.append(d.toString()));
    proc.stderr?.on("data", (d: Buffer) => this.output.append(d.toString()));
    proc.on("error", (err) => {
      this.output.appendLine(`Could not start Python: ${err.message}`);
      if (this.proc === proc) {
        this.proc = undefined;
        this.set("offline", `Couldn't start Python ("${python}"). Set "pyv.pythonPath" in VS Code settings.`);
      }
    });
    proc.on("exit", (code) => {
      this.output.appendLine(`V's server exited (code ${code ?? "none"}).`);
      if (this.proc === proc) {
        this.proc = undefined;
        this.set("offline", "V's server stopped unexpectedly — see Output → V Server.");
      }
    });
  }

  /** Stop the server this extension started (never one started elsewhere). */
  stop(reason: string): void {
    this.cancelStop();
    const proc = this.proc;
    if (!proc) {
      return;
    }
    this.proc = undefined;
    this.output.appendLine(`Stopping V's server (${reason}).`);
    killTree(proc);
    this.set("offline");
  }

  dispose(): void {
    this.stopPolling();
    this.stop("VS Code closing");
    this.output.dispose();
  }

  // ─── internals ──────────────────────────────────────────────────────────────

  private set(state: ServerState, detail?: string): void {
    if (state === this.state && detail === this.detail) {
      return;
    }
    this.state = state;
    this.detail = detail;
    for (const listener of this.listeners) {
      listener(state, detail);
    }
  }

  // While the panel is visible: online as soon as the server answers — ours
  // after loading, or one started in a terminal
  private startPolling(): void {
    if (this.pollTimer) {
      return;
    }
    this.pollTimer = setInterval(async () => {
      const up = await isUp();
      if (up) {
        this.set("online");
      } else if (this.state === "online") {
        this.set(this.proc ? "starting" : "offline");
      }
    }, POLL_MS);
  }

  private stopPolling(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = undefined;
    }
  }

  private cancelStop(): void {
    if (this.stopTimer) {
      clearTimeout(this.stopTimer);
      this.stopTimer = undefined;
    }
  }
}

function settings() {
  const cfg = vscode.workspace.getConfiguration("pyv");
  const url = new URL(cfg.get<string>("serverUrl", "http://localhost:8000"));
  return {
    manageServer: cfg.get<boolean>("manageServer", true),
    stopAfterSeconds: cfg.get<number>("stopServerAfterSeconds", 120),
    python: cfg.get<string>("pythonPath", "python") || "python",
    projectPath: cfg.get<string>("projectPath", ""),
    host: url.hostname === "localhost" ? "127.0.0.1" : url.hostname,
    port: url.port || "8000",
  };
}

async function isUp(): Promise<boolean> {
  try {
    await checkHealth();
    return true;
  } catch {
    return false;
  }
}

function isPyV(folder: string): boolean {
  return fs.existsSync(path.join(folder, "inference", "api", "main.py"));
}

/**
 * The Py-V folder to start the server in: the "pyv.projectPath" setting, else
 * an open workspace folder that has inference/api/main.py — only a trusted
 * one, since starting the server runs that folder's code.
 */
function findProject(): { folder?: string; problem?: string } {
  const configured = settings().projectPath;
  if (configured) {
    return isPyV(configured)
      ? { folder: configured }
      : { problem: `"pyv.projectPath" (${configured}) has no inference/api/main.py.` };
  }
  const folder = (vscode.workspace.workspaceFolders ?? []).map((f) => f.uri.fsPath).find(isPyV);
  if (!folder) {
    return { problem: 'Open the Py-V folder, or set "pyv.projectPath", so I can start my server.' };
  }
  if (!vscode.workspace.isTrusted) {
    return { problem: "Trust this folder (it runs my server's code), then press Start V." };
  }
  return { folder };
}

function killTree(proc: ChildProcess): void {
  try {
    if (process.platform === "win32" && proc.pid) {
      // Synchronous, so it also finishes while VS Code is closing; /T takes its child processes too
      execFileSync("taskkill", ["/pid", String(proc.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
    } else {
      proc.kill();
    }
  } catch {
    // already gone
  }
}
