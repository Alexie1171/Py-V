/**
 * file_context.ts — PY-V VS Code Extension (Phase 10.2)
 * The file open in the editor, for the chat panel: what the panel shows
 * ("app.py · lines 10-24") and what goes to the server with a message. The
 * server decides how much of it goes into the prompt
 * (inference/engine/file_context.py — when the message is about the file, a
 * selection always, a long file in pieces).
 *
 * Only real documents count: files on disk, unsaved "Untitled" files and
 * notebook cells — not the Output panel, git views or settings.
 */

import * as vscode from "vscode";

/** What the server gets (inference/api/schemas.OpenFile). */
export interface OpenFilePayload {
  name: string;
  language_id: string;
  content: string;
  first_line: number;
  selection: string;
  selection_line: number;
  cursor_line: number;
}

/** What the panel shows about the open file. */
export interface EditorInfo {
  name: string;
  where: string | null; // "lines 10-24" / "line 7" when something is selected
}

const READABLE = new Set(["file", "untitled", "vscode-notebook-cell"]);
const MAX_SEND = 300_000; // characters of a file sent at most — a bigger file goes as a window around the cursor

export function isReadable(editor: vscode.TextEditor | undefined): editor is vscode.TextEditor {
  return !!editor && READABLE.has(editor.document.uri.scheme) && !editor.document.isClosed;
}

function fileName(doc: vscode.TextDocument): string {
  if (doc.uri.scheme === "untitled") {
    return doc.fileName;
  }
  const uri = doc.uri.scheme === "vscode-notebook-cell" ? doc.uri.with({ scheme: "file", fragment: "" }) : doc.uri;
  return vscode.workspace.asRelativePath(uri, false);
}

function selectedLines(editor: vscode.TextEditor): { first: number; last: number } | null {
  const sel = editor.selection;
  if (sel.isEmpty) {
    return null;
  }
  // A selection ending at the start of a line doesn't include that line
  const last = sel.end.character === 0 && sel.end.line > sel.start.line ? sel.end.line - 1 : sel.end.line;
  return { first: sel.start.line + 1, last: last + 1 };
}

export function describeEditor(editor: vscode.TextEditor): EditorInfo {
  const lines = selectedLines(editor);
  const where = !lines ? null : lines.first === lines.last ? `line ${lines.first}` : `lines ${lines.first}-${lines.last}`;
  const name = fileName(editor.document).split(/[\\/]/).pop() || fileName(editor.document);
  return { name, where };
}

export function readOpenFile(editor: vscode.TextEditor): OpenFilePayload {
  const doc = editor.document;
  const cursor = editor.selection.active.line;
  const lines = selectedLines(editor);

  let content = doc.getText();
  let firstLine = 1;
  if (content.length > MAX_SEND) {
    // A huge file: the lines around the cursor, as many as fit
    let lo = cursor;
    let hi = cursor;
    let size = doc.lineAt(cursor).text.length + 1;
    let grew = true;
    while (grew) {
      grew = false;
      if (hi + 1 < doc.lineCount && size + doc.lineAt(hi + 1).text.length + 1 <= MAX_SEND) {
        hi += 1;
        size += doc.lineAt(hi).text.length + 1;
        grew = true;
      }
      if (lo > 0 && size + doc.lineAt(lo - 1).text.length + 1 <= MAX_SEND) {
        lo -= 1;
        size += doc.lineAt(lo).text.length + 1;
        grew = true;
      }
    }
    content = doc.getText(new vscode.Range(lo, 0, hi, doc.lineAt(hi).text.length));
    firstLine = lo + 1;
  }

  return {
    name: fileName(doc),
    language_id: doc.languageId,
    content,
    first_line: firstLine,
    selection: lines ? doc.getText(editor.selection).slice(0, MAX_SEND) : "",
    selection_line: lines ? lines.first : 0,
    cursor_line: cursor + 1,
  };
}
