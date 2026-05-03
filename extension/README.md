# PY-V VS Code Extension

Local AI-powered Python code generation inside VS Code, backed by your fine-tuned Phi-2 model.

---

## Requirements

- The PY-V inference server must be running locally
- Node.js 18+
- VS Code 1.85+

---

## Setup

### 1. Start the inference server

In your PY-V project directory:

```bash
uvicorn inference.api.main:app --host 0.0.0.0 --port 8000
```

Wait for `Server ready.` before using the extension.

### 2. Install dependencies and compile

```bash
cd extension
npm install
npm run compile
```

### 3. Run in VS Code

Press `F5` in VS Code with the extension folder open to launch a development instance with PY-V loaded.

---

## Usage

### Generate from a comment (recommended)

Write a comment describing what you want, place your cursor on it, and press `Ctrl+Shift+G`:

```python
# Write a function to check if a number is prime
```

The generated code will be inserted directly below.

### Generate from a typed prompt

Press `Ctrl+Shift+P` and type your instruction in the input box that appears.

### Generate from selected text

Select any text in a Python file and press `Ctrl+Shift+G`. The selected text will be used as the instruction and replaced with the generated code.

### Check server status

Click the `⟡ PY-V` item in the status bar (bottom right) or run `PY-V: Check Server Status` from the command palette. You'll see a confirmation with the model name if the server is reachable.

---

## Keybindings

| Action | Shortcut |
|--------|----------|
| Generate from selection / comment | `Ctrl+Shift+G` |
| Generate from typed prompt | `Ctrl+Shift+P` |

---

## Configuration

Open VS Code settings and search for `PY-V`:

| Setting | Default | Description |
|---------|---------|-------------|
| `pyv.serverUrl` | `http://localhost:8000` | Inference server URL |
| `pyv.maxTokens` | `256` | Max tokens to generate |
| `pyv.temperature` | `0.2` | Sampling temperature |

---

## Troubleshooting

**"Cannot reach server"** — Make sure uvicorn is running and the port matches `pyv.serverUrl` in settings.

**Empty output** — Try rephrasing your prompt to be more specific, e.g. `Write a Python function that...`.

**Slow generation** — Normal on GTX 1650. Expect 10–30 seconds depending on `maxTokens`.