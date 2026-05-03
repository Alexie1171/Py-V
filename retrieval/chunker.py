# retrieval/chunker.py

import ast
from typing import List, Dict


def chunk_python_file(code: str, file_path: str) -> List[Dict]:
    chunks = []

    try:
        tree = ast.parse(code)
    except Exception:
        return chunks  # skip invalid files

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                chunk_code = ast.get_source_segment(code, node)
                if not chunk_code:
                    continue

                chunks.append({
                    "content": chunk_code,
                    "metadata": {
                        "file": file_path,
                        "name": getattr(node, "name", "unknown"),
                        "type": type(node).__name__,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                    }
                })
            except Exception:
                continue

    return chunks