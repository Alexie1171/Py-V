/**
 * provider.ts — PY-V VS Code Extension
 * Handles inserting generated code into the active editor.
 * Keeps all editor manipulation logic separate from command logic.
 */

import * as vscode from "vscode";

/**
 * Insert generated code at the current cursor position,
 * or replace the current selection if one exists.
 */
export async function insertCode(
  editor: vscode.TextEditor,
  code: string
): Promise<void> {
  const selection = editor.selection;

  await editor.edit((editBuilder) => {
    if (!selection.isEmpty) {
      // Replace selected text (e.g. a comment used as a prompt)
      editBuilder.replace(selection, code);
    } else {
      // Insert at cursor position
      editBuilder.insert(selection.active, "\n" + code);
    }
  });

  // Move cursor to end of inserted code
  const newPosition = editor.selection.end;
  editor.selection = new vscode.Selection(newPosition, newPosition);
  editor.revealRange(new vscode.Range(newPosition, newPosition));
}

/**
 * Extract a meaningful instruction from the editor context.
 * Priority: selected text → comment on current line → line content.
 */
export function extractInstruction(editor: vscode.TextEditor): string {
  const selection = editor.selection;
  const document = editor.document;

  // Use selected text if it exists
  if (!selection.isEmpty) {
    return document.getText(selection).trim();
  }

  // Check if current line is a comment — use it as the instruction
  const currentLine = document.lineAt(selection.active.line).text.trim();
  if (currentLine.startsWith("#")) {
    return currentLine.replace(/^#+\s*/, "").trim();
  }

  return "";
}