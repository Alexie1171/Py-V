/**
 * extension.ts — PY-V VS Code Extension
 * Main entry point. Registers all commands and handles activation/deactivation.
 *
 * Commands:
 *   pyv.generate          — generate from selected text or comment (Ctrl+Shift+G)
 *   pyv.generateFromInput — generate from a typed prompt (Ctrl+Shift+P)
 *   pyv.checkServer       — ping the inference server
 */

import * as vscode from "vscode";
import { generateCode, checkHealth } from "./api";
import { insertCode, extractInstruction } from "./provider";

// ─── Activation ───────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext): void {
  console.log("PY-V extension activated.");

  // Show status bar item
  const statusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBar.text = "$(sparkle) PY-V";
  statusBar.tooltip = "PY-V: Local Python AI Assistant";
  statusBar.command = "pyv.checkServer";
  statusBar.show();
  context.subscriptions.push(statusBar);

  // ── Command: Generate from selection or comment ──────────────────────────

  const generateCmd = vscode.commands.registerCommand(
    "pyv.generate",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("PY-V: No active editor.");
        return;
      }

      const instruction = extractInstruction(editor);
      if (!instruction) {
        vscode.window.showWarningMessage(
          "PY-V: Select text or place cursor on a comment to use as a prompt."
        );
        return;
      }

      await runGeneration(editor, instruction, statusBar);
    }
  );

  // ── Command: Generate from typed prompt ──────────────────────────────────

  const generateFromInputCmd = vscode.commands.registerCommand(
    "pyv.generateFromInput",
    async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showWarningMessage("PY-V: No active editor.");
        return;
      }

      const instruction = await vscode.window.showInputBox({
        prompt: "PY-V: Describe the Python code you want to generate",
        placeHolder: "e.g. Write a function to sort a list of dictionaries by key",
        ignoreFocusOut: true,
      });

      if (!instruction || instruction.trim() === "") {
        return; // user cancelled
      }

      await runGeneration(editor, instruction.trim(), statusBar);
    }
  );

  // ── Command: Check server health ─────────────────────────────────────────

  const checkServerCmd = vscode.commands.registerCommand(
    "pyv.checkServer",
    async () => {
      try {
        const health = await checkHealth();
        vscode.window.showInformationMessage(
          `PY-V: Server online ✔ — model: ${health.model}`
        );
        statusBar.text = "$(sparkle) PY-V ✔";
      } catch {
        vscode.window.showErrorMessage(
          "PY-V: Server unreachable. Make sure uvicorn is running on port 8000."
        );
        statusBar.text = "$(sparkle) PY-V ✖";
      }
    }
  );

  context.subscriptions.push(generateCmd, generateFromInputCmd, checkServerCmd);
}

// ─── Deactivation ─────────────────────────────────────────────────────────────

export function deactivate(): void {
  console.log("PY-V extension deactivated.");
}

// ─── Shared Generation Flow ───────────────────────────────────────────────────

async function runGeneration(
  editor: vscode.TextEditor,
  instruction: string,
  statusBar: vscode.StatusBarItem
): Promise<void> {
  statusBar.text = "$(loading~spin) PY-V generating...";

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: `PY-V: Generating code...`,
      cancellable: false,
    },
    async () => {
      try {
        const code = await generateCode(instruction);

        if (!code || code.trim() === "") {
          vscode.window.showWarningMessage("PY-V: Model returned empty output.");
          return;
        }

        await insertCode(editor, code);
        statusBar.text = "$(sparkle) PY-V ✔";
        vscode.window.showInformationMessage("PY-V: Code inserted ✔");
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);

        if (message.includes("ECONNREFUSED") || message.includes("timed out")) {
          vscode.window.showErrorMessage(
            "PY-V: Cannot reach server. Is uvicorn running on port 8000?"
          );
        } else {
          vscode.window.showErrorMessage(`PY-V: Generation failed — ${message}`);
        }

        statusBar.text = "$(sparkle) PY-V ✖";
      }
    }
  );
}