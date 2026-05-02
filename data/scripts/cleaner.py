"""
cleaner.py — PY-V Data Pipeline
Upgraded v2: AST validation, encoding fixes, length bounds,
             whitespace normalization, quality heuristics.
"""

import re
import ast
import logging
from html import unescape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MIN_CODE_LENGTH  = 50       # chars — below this is almost certainly noise
MAX_CODE_LENGTH  = 8000     # chars — above this is too large for training
MIN_LINE_COUNT   = 3        # lines
MAX_LINE_COUNT   = 200      # lines

PYTHON_SIGNALS   = [
    "def ", "class ", "import ", "return ",
    "self.", "print(", "if __name__",
]

NOISE_PATTERNS   = [
    r"^#+\s",                   # markdown headers
    r"^\s*```",                 # markdown fences
    r"^\s*\.\.\.",              # ellipsis-only lines
    r"^(TODO|FIXME|HACK|XXX)",  # unfinished stubs
]

# ─── Text Cleaning ────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text)


def fix_encoding(text: str) -> str:
    """Fix common encoding artifacts."""
    replacements = {
        "\u2019": "'",   # right single quote
        "\u2018": "'",   # left single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\r\n":   "\n",  # Windows line endings
        "\r":     "\n",  # old Mac line endings
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize indentation to 4-space standard.
    Strips trailing whitespace per line, removes excessive blank lines.
    """
    lines    = text.splitlines()
    cleaned  = [line.rstrip() for line in lines]

    # Collapse 3+ consecutive blank lines into 2
    result   = []
    blanks   = 0
    for line in cleaned:
        if line == "":
            blanks += 1
            if blanks <= 2:
                result.append(line)
        else:
            blanks = 0
            result.append(line)

    return "\n".join(result).strip()


def clean_code(text: str) -> str:
    """Full cleaning pipeline for a single code sample."""
    text = strip_html(text)
    text = fix_encoding(text)
    text = normalize_whitespace(text)
    return text


def clean_instruction(text: str) -> str:
    """Clean a natural-language instruction string."""
    text = strip_html(text)
    text = fix_encoding(text)
    text = re.sub(r"\s+", " ", text)   # collapse internal whitespace
    return text.strip()

# ─── Validation ───────────────────────────────────────────────────────────────

def has_python_signals(code: str) -> bool:
    """Check if code looks like Python (not SQL, JS, shell, etc.)."""
    return any(signal in code for signal in PYTHON_SIGNALS)


def is_parseable(code: str) -> bool:
    """Try to parse with AST. Rejects syntactically broken code."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def has_noise_lines(code: str) -> bool:
    """Return True if code contains markdown or stub noise."""
    for line in code.splitlines():
        for pattern in NOISE_PATTERNS:
            if re.match(pattern, line):
                return True
    return False


def is_valid_sample(sample: dict) -> tuple[bool, str]:
    """
    Validate a training sample dict with keys 'instruction' and 'output'.
    Returns (is_valid, reason_if_rejected).
    """
    code        = sample.get("output", "")
    instruction = sample.get("instruction", "")

    if not code or not instruction:
        return False, "missing instruction or output"

    if len(code) < MIN_CODE_LENGTH:
        return False, f"code too short ({len(code)} chars)"

    if len(code) > MAX_CODE_LENGTH:
        return False, f"code too long ({len(code)} chars)"

    lines = code.splitlines()
    if len(lines) < MIN_LINE_COUNT:
        return False, f"too few lines ({len(lines)})"

    if len(lines) > MAX_LINE_COUNT:
        return False, f"too many lines ({len(lines)})"

    if not has_python_signals(code):
        return False, "no Python signals detected"

    if not is_parseable(code):
        return False, "AST parse failed (syntax error)"

    if has_noise_lines(code):
        return False, "contains markdown or stub noise"

    if len(instruction.strip()) < 10:
        return False, "instruction too short"

    return True, ""

# ─── Batch Cleaning ───────────────────────────────────────────────────────────

def clean_dataset(samples: list[dict]) -> list[dict]:
    """
    Run the full clean + validate pipeline over a list of samples.
    Returns only valid, cleaned samples.
    """
    cleaned  = []
    rejected = 0

    for sample in samples:
        # Clean fields in place
        sample["output"]      = clean_code(sample.get("output", ""))
        sample["instruction"] = clean_instruction(sample.get("instruction", ""))

        valid, reason = is_valid_sample(sample)

        if valid:
            cleaned.append(sample)
        else:
            rejected += 1
            logger.debug(f"Rejected: {reason} | {sample.get('instruction', '')[:60]}")

    logger.info(f"Cleaning complete: {len(cleaned)} kept, {rejected} rejected")
    return cleaned