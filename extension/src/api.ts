/**
 * api.ts — PY-V VS Code Extension
 * Typed HTTP client for the PY-V FastAPI inference server.
 * All server communication goes through this module.
 */

import * as vscode from "vscode";
import * as http from "http";
import { OpenFilePayload } from "./file_context";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface GenerateRequest {
  instruction: string;
  max_tokens: number;
  temperature: number;
}

export interface GenerateResponse {
  instruction: string;
  code: string;
  tokens_used: number;
}

export interface HealthResponse {
  status: string;
  model: string;
}

// ─── Config Helpers ───────────────────────────────────────────────────────────

function getConfig() {
  const cfg = vscode.workspace.getConfiguration("pyv");
  return {
    serverUrl: cfg.get<string>("serverUrl", "http://localhost:8000"),
    maxTokens: cfg.get<number>("maxTokens", 256),
    temperature: cfg.get<number>("temperature", 0.2),
  };
}

// ─── HTTP Helper ──────────────────────────────────────────────────────────────

function postJson<T>(url: string, body: object): Promise<T> {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body);
    const parsed = new URL(url);

    const options: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port || 80,
      path: parsed.pathname,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
      },
    };

    const req = http.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data) as T);
        } catch {
          reject(new Error(`Failed to parse response: ${data}`));
        }
      });
    });

    req.on("error", (err) => reject(err));
    req.setTimeout(60000, () => {
      req.destroy();
      reject(new Error("Request timed out after 60s"));
    });

    req.write(payload);
    req.end();
  });
}

function getJson<T>(url: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);

    const options: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port || 80,
      path: parsed.pathname,
      method: "GET",
    };

    const req = http.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          resolve(JSON.parse(data) as T);
        } catch {
          reject(new Error(`Failed to parse response: ${data}`));
        }
      });
    });

    req.on("error", (err) => reject(err));
    req.setTimeout(5000, () => {
      req.destroy();
      reject(new Error("Health check timed out"));
    });

    req.end();
  });
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Generate Python code from a natural language instruction.
 * Returns the generated code string, or throws on failure.
 */
export async function generateCode(instruction: string): Promise<string> {
  const { serverUrl, maxTokens, temperature } = getConfig();

  const response = await postJson<GenerateResponse>(
    `${serverUrl}/api/v1/generate`,
    {
      instruction,
      max_tokens: maxTokens,
      temperature,
    }
  );

  return response.code;
}

// ─── Chat stream (chat panel) ─────────────────────────────────────────────────

export interface ChatDone {
  session_id?: string;
  response: string;
  mode: string;
  confidence: number;
  rag_chunks: number;
  memories: number;
  language: string | null;
  load: string | null;
  note: string | null;
  file: string | null; // what of the open file V read ("app.py, lines 10-24")
  notes?: number; // study notes used (Phase 13)
  project?: number; // project pieces used (10.5)
  project_files?: string[];
  sources?: { title: string; url: string }[] | null; // a web lookup's pages
  asks?: string[] | null; // quick replies ("Yes, look it up" / "No thanks")
  study?: StudyStatus | null;
}

export interface StudyStatus {
  running: boolean;
  topic?: string;
  minutes?: number;
  minutes_left?: number;
  notes?: number;
  state?: string;
  paused?: boolean;
}

/** What goes with a message besides its text. */
export interface ChatExtras {
  file?: OpenFilePayload; // the open file (10.2)
  project?: string; // the workspace folder (10.5)
}

export type ChatEvent =
  | { kind: "start"; mode: string; confidence: number; load: string | null; language: string | null; file: string | null }
  | { kind: "status"; text: string } // she is looking something up online (Phase 13)
  | { kind: "piece"; text: string }
  | { kind: "done"; result: ChatDone }
  | { kind: "error"; detail: string };

/**
 * Send a chat message and receive V's answer as it is written
 * (POST /api/v1/chat/stream, server-sent events). extras: the open file
 * (file_context.ts; none = not shared) and the workspace folder (project
 * search). onEvent gets start → status… → piece… → done, or error. Returns a function that stops the answer —
 * closing the connection makes the server stop the brain.
 *
 * Uses fetch and reads the body itself: with Node's http module the event
 * handling ran inside the HTTP parser, and in VS Code's extension host that
 * ended in "Parse Error: JS Exception" (2026-09-26, first panel try).
 */
export function chatStream(
  sessionId: string,
  message: string,
  extras: ChatExtras,
  onEvent: (event: ChatEvent) => void
): () => void {
  const { serverUrl } = getConfig();
  const controller = new AbortController();
  let finished = false;

  const emit = (event: ChatEvent) => {
    if (finished) {
      return;
    }
    if (event.kind === "done" || event.kind === "error") {
      finished = true;
    }
    try {
      onEvent(event);
    } catch (err) {
      // A broken handler must not kill the stream silently — say what broke
      console.error("V chat panel:", err);
      if (event.kind !== "error") {
        finished = true;
        onEvent({ kind: "error", detail: `Panel error: ${describeError(err)}` });
      }
    }
  };

  (async () => {
    let res: Response;
    try {
      res = await fetch(`${serverUrl}/api/v1/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message, file: extras.file, project: extras.project }),
        signal: controller.signal,
      });
    } catch (err) {
      if (!controller.signal.aborted) {
        emit({ kind: "error", detail: describeError(err) });
      }
      return;
    }
    if (!res.ok || !res.body) {
      emit({ kind: "error", detail: `Server answered ${res.status}` });
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        let end: number;
        while ((end = buffer.indexOf("\n\n")) !== -1) {
          const event = parseSse(buffer.slice(0, end));
          buffer = buffer.slice(end + 2);
          if (event) {
            emit(event);
          }
        }
      }
      emit({ kind: "error", detail: "The answer ended early." });
    } catch (err) {
      if (!controller.signal.aborted) {
        emit({ kind: "error", detail: describeError(err) });
      }
    }
  })();

  return () => {
    finished = true;
    controller.abort();
  };
}

/**
 * An error as text, including what's inside it: "can't connect" arrives as
 * "fetch failed" around one ECONNREFUSED per address (IPv6 and IPv4).
 */
export function describeError(err: unknown): string {
  const parts: string[] = [];
  const visit = (e: any) => {
    if (!e) {
      return;
    }
    if (e.code) {
      parts.push(String(e.code));
    } else if (e.message) {
      parts.push(String(e.message));
    }
    if (Array.isArray(e.errors)) {
      e.errors.forEach(visit);
    }
    if (e.cause) {
      visit(e.cause);
    }
  };
  visit(err);
  return [...new Set(parts)].join(", ") || String(err);
}

function parseSse(block: string): ChatEvent | null {
  let name = "";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      name = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).trimStart());
    }
  }
  if (!name || data.length === 0) {
    return null;
  }
  let body: any;
  try {
    body = JSON.parse(data.join("\n"));
  } catch {
    return null;
  }
  switch (name) {
    case "start":
      return {
        kind: "start",
        mode: body.mode,
        confidence: body.confidence,
        load: body.load ?? null,
        language: body.language ?? null,
        file: body.file ?? null,
      };
    case "status":
      return { kind: "status", text: body.text ?? "" };
    case "piece":
      return { kind: "piece", text: body.text ?? "" };
    case "done":
      return { kind: "done", result: body as ChatDone };
    case "error":
      return { kind: "error", detail: body.detail ?? "Unknown error" };
    default:
      return null;
  }
}

// ─── Small JSON calls (fetch) ─────────────────────────────────────────────────

/** A JSON request to the server; throws with the server's own reason ("detail") when it says no. */
async function requestJson<T>(method: string, path: string, body?: object): Promise<T> {
  const { serverUrl } = getConfig();
  let res: Response;
  try {
    res = await fetch(`${serverUrl}/api/v1${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new Error(describeError(err));
  }
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    // not JSON — reported below
  }
  if (!res.ok) {
    const detail = typeof data?.detail === "string" ? data.detail : `Server answered ${res.status}`;
    throw new Error(detail);
  }
  return data as T;
}

export interface ChangeProposal {
  content: string;
  how: string;
  summary: string;
}

/** Where a code block goes in the file (Phase 10.3) — the whole file with the change; nothing is written. */
export function proposeChange(code: string, file: OpenFilePayload): Promise<ChangeProposal> {
  return requestJson<ChangeProposal>("POST", "/file/change", { code, file });
}

export interface MemoryFact {
  id: number;
  key: string;
  text: string;
  created: number;
}

export interface MemoryList {
  enabled: boolean;
  facts: MemoryFact[];
  messages: number;
  sessions: number;
}

/** What V remembers (Phase 10.4 memory view). */
export function getMemory(): Promise<MemoryList> {
  return requestJson<MemoryList>("GET", "/memory");
}

/** Make V forget one fact for good. */
export function forgetFact(id: number): Promise<unknown> {
  return requestJson("DELETE", `/memory/${id}`);
}

// ─── Project search (10.5) ────────────────────────────────────────────────────

export interface ProjectStatus {
  root: string;
  status: string;
  files?: number;
  pieces?: number;
  embedded?: number;
}

/** Index (or bring up to date) the workspace folder — in the background on the server; stays local. */
export function indexProject(root: string): Promise<ProjectStatus> {
  return requestJson<ProjectStatus>("POST", "/project/index", { root });
}

export function projectStatus(root: string): Promise<ProjectStatus> {
  return requestJson<ProjectStatus>("GET", `/project/status?root=${encodeURIComponent(root)}`);
}

// ─── Learning (Phase 13) ──────────────────────────────────────────────────────

export interface LearningOverview {
  topics: any[];
  approved: number;
  study: StudyStatus | null;
}

export function getLearning(): Promise<LearningOverview> {
  return requestJson<LearningOverview>("GET", "/learning");
}

export function approveAnswer(question: string, answer: string, mode: string | null, language: string | null,
                              sessionId: string): Promise<{ id: string }> {
  return requestJson("POST", "/learning/approve", { question, answer, mode, language, session_id: sessionId });
}

export function unapproveAnswer(id: string): Promise<unknown> {
  return requestJson("DELETE", `/learning/approve/${encodeURIComponent(id)}`);
}

export function approveTopic(id: number, approved: boolean): Promise<unknown> {
  return requestJson("POST", `/learning/topics/${id}/approve`, { approved });
}

export function forgetTopic(id: number): Promise<unknown> {
  return requestJson("DELETE", `/learning/topics/${id}`);
}

export function studyStatus(): Promise<StudyStatus> {
  return requestJson<StudyStatus>("GET", "/learning/study");
}

export function stopStudy(): Promise<{ reply: string }> {
  return requestJson("POST", "/learning/study/stop");
}

/**
 * Ping the server health endpoint.
 * Returns the HealthResponse, or throws if the server is unreachable.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const { serverUrl } = getConfig();
  return getJson<HealthResponse>(`${serverUrl}/api/v1/health`);
}