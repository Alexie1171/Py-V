"""
code_runner.py — PY-V (experiments/)
Runs a Python program in a separate process with a timeout and a guard that
disables the most destructive calls (deleting/renaming files, writing files,
starting processes, network sockets).

Best-effort guard, NOT a real sandbox. Fine for short model-written functions
checked by asserts; untrusted code from the internet should run on Colab.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Prepended to every program before it runs.
_GUARD = '''
import builtins, os, shutil, socket, subprocess

for _name in ["system", "remove", "removedirs", "rmdir", "unlink", "rename",
              "renames", "replace", "truncate", "chmod", "chown", "kill",
              "killpg", "fork", "forkpty", "putenv", "chdir", "fchdir"]:
    if hasattr(os, _name):
        setattr(os, _name, None)

shutil.rmtree = shutil.move = shutil.chown = None
subprocess.Popen = None
socket.socket = None
builtins.exit = builtins.quit = None

_real_open = builtins.open

def _read_only_open(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in "wax+"):
        raise PermissionError("writing files is disabled by code_runner")
    return _real_open(file, mode, *args, **kwargs)

builtins.open = _read_only_open
del _name
'''


def run_python(program: str, timeout: int) -> tuple:
    """
    Run `program` in a fresh Python process inside a temporary directory.

    Returns:
        (passed, error) — passed is True when the process exits with code 0
        within `timeout` seconds; error is the last stderr line otherwise.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(_GUARD + "\n" + program, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd            = tmp,
                capture_output = True,
                text           = True,
                timeout        = timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"

    if result.returncode == 0:
        return True, ""

    lines = [line for line in result.stderr.strip().split("\n") if line.strip()]
    return False, lines[-1] if lines else f"exit code {result.returncode}"
