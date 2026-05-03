/**
 * api.ts — PY-V VS Code Extension
 * Typed HTTP client for the PY-V FastAPI inference server.
 * All server communication goes through this module.
 */

import * as vscode from "vscode";
import * as http from "http";

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

/**
 * Ping the server health endpoint.
 * Returns the HealthResponse, or throws if the server is unreachable.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const { serverUrl } = getConfig();
  return getJson<HealthResponse>(`${serverUrl}/api/v1/health`);
}