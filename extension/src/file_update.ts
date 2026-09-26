/**
 * file_update.ts — PY-V VS Code Extension (Phase 10.3)
 * "Apply to file" on a code block of V's answer: the server works out where it
 * goes (POST /api/v1/file/change — inference/engine/file_update.py), VS Code's
 * diff view shows the file with the change, and the file is changed only when
 * the user clicks Apply in the panel (owner's rule: never on her own).
 * Discard closes the diff and changes nothing. Applied changes are left
 * unsaved — Ctrl+S saves, Ctrl+Z undoes.
 *
 * The proposed file is served from memory under the pyv-change: scheme; one
 * proposal per code block (key from the page).
 */

import * as vscode from "vscode";
import { proposeChange } from "./api";
import { readOpenFile } from "./file_context";

/** The file an answer read, saved with the answer (the page keeps it). */
export interface Target {
  uri: string; // document URI
  name: string; // relative name, for messages
  selection: string; // what was selected when the question was sent ("" = nothing)
  selectionLine: number;
}

interface Proposal {
  docUri: vscode.Uri;
  name: string;
  version: number; // the document's version the change was worked out on
  code: string;
  target?: Target;
  content: string;
  summary: string;
  proposedUri: vscode.Uri;
}

export class ChangeReviewer implements vscode.TextDocumentContentProvider, vscode.Disposable {
  static readonly scheme = "pyv-change";

  private readonly proposals = new Map<string, Proposal>();
  private readonly changed = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this.changed.event;
  private readonly registration = vscode.workspace.registerTextDocumentContentProvider(ChangeReviewer.scheme, this);

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.proposals.get(uri.query)?.content ?? "";
  }

  /** Work out the change and open the diff. Returns what the change does ("replaces add()"). */
  async show(key: string, code: string, target: Target | undefined, fallback: vscode.TextEditor | undefined): Promise<string> {
    const doc = await this.document(target, fallback);
    const name = target?.name ?? vscode.workspace.asRelativePath(doc.uri, false);
    const proposal: Proposal = {
      docUri: doc.uri,
      name,
      version: doc.version,
      code,
      target,
      content: "",
      summary: "",
      proposedUri: vscode.Uri.from({ scheme: ChangeReviewer.scheme, path: "/" + baseName(name), query: key }),
    };
    await this.work(proposal, doc);
    this.proposals.set(key, proposal);
    this.changed.fire(proposal.proposedUri);
    await vscode.commands.executeCommand("vscode.diff", doc.uri, proposal.proposedUri, `${baseName(name)}: V's change`, {
      preview: true,
    });
    return proposal.summary;
  }

  /** Write the change into the file (unsaved). If the file changed since the diff, the change is worked out again first. */
  async apply(key: string): Promise<{ ok: boolean; text: string }> {
    const p = this.proposals.get(key);
    if (!p) {
      return { ok: false, text: "That change isn't open any more - click Apply to file again." };
    }
    const doc = await vscode.workspace.openTextDocument(p.docUri);
    if (doc.version !== p.version) {
      await this.work(p, doc);
      this.changed.fire(p.proposedUri);
      return { ok: false, text: `${baseName(p.name)} changed since - I updated the diff. Check it and click Apply again.` };
    }
    const old = doc.getText();
    let start = 0;
    while (start < old.length && start < p.content.length && old[start] === p.content[start]) {
      start++;
    }
    let end = 0;
    while (
      end < old.length - start &&
      end < p.content.length - start &&
      old[old.length - 1 - end] === p.content[p.content.length - 1 - end]
    ) {
      end++;
    }
    const edit = new vscode.WorkspaceEdit();
    const range = new vscode.Range(doc.positionAt(start), doc.positionAt(old.length - end));
    edit.replace(doc.uri, range, p.content.slice(start, p.content.length - end));
    if (!(await vscode.workspace.applyEdit(edit))) {
      return { ok: false, text: `VS Code refused the edit to ${baseName(p.name)}.` };
    }
    await this.close(key);
    const editor = await vscode.window.showTextDocument(doc, { preview: false });
    const at = doc.positionAt(start);
    editor.selection = new vscode.Selection(at, at);
    editor.revealRange(new vscode.Range(at, at), vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    return { ok: true, text: `Applied to ${baseName(p.name)} (not saved yet - Ctrl+S saves, Ctrl+Z undoes).` };
  }

  async discard(key: string): Promise<void> {
    await this.close(key);
  }

  dispose(): void {
    this.registration.dispose();
    this.changed.dispose();
    this.proposals.clear();
  }

  // ─── internals ─────────────────────────────────────────────────────────────

  private async document(target: Target | undefined, fallback: vscode.TextEditor | undefined): Promise<vscode.TextDocument> {
    if (target?.uri) {
      try {
        return await vscode.workspace.openTextDocument(vscode.Uri.parse(target.uri));
      } catch {
        throw new Error(`I can't open ${target.name} any more.`);
      }
    }
    if (fallback) {
      return fallback.document;
    }
    throw new Error("Open the file the code is for, then click Apply to file again.");
  }

  /** Ask the server where the code goes, on the document as it is now. */
  private async work(p: Proposal, doc: vscode.TextDocument): Promise<void> {
    const editor = vscode.window.visibleTextEditors.find((e) => e.document === doc);
    const file = editor
      ? readOpenFile(editor)
      : { name: p.name, language_id: doc.languageId, content: doc.getText(), first_line: 1, selection: "", selection_line: 0, cursor_line: 0 };
    file.content = doc.getText(); // the whole file, even a huge one
    file.first_line = 1;
    file.selection = p.target?.selection ?? ""; // what the answer read, not what is selected now
    file.selection_line = p.target?.selectionLine ?? 0;
    const change = await proposeChange(p.code, file);
    p.content = change.content;
    p.summary = change.summary;
    p.version = doc.version;
  }

  private async close(key: string): Promise<void> {
    const p = this.proposals.get(key);
    this.proposals.delete(key);
    if (!p) {
      return;
    }
    const tabs = vscode.window.tabGroups.all.flatMap((g) => g.tabs);
    for (const tab of tabs) {
      if (tab.input instanceof vscode.TabInputTextDiff && tab.input.modified.toString() === p.proposedUri.toString()) {
        await vscode.window.tabGroups.close(tab);
      }
    }
  }
}

function baseName(name: string): string {
  return name.split(/[\\/]/).pop() || name;
}
