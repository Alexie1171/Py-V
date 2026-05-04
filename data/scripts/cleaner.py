"""
cleaner.py — PY-V Data Pipeline
Upgraded v3: Tighter quality gates — higher min length, stricter Python
             signal requirements, docstring bonus, more noise patterns.
"""

import re
import ast
import logging
from html import unescape

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MIN_CODE_LENGTH  = 100      # was 50 — filters out trivial one-liners
MAX_CODE_LENGTH  = 8000     # chars
MIN_LINE_COUNT   = 5        # was 3 — requires at least a meaningful body
MAX_LINE_COUNT   = 200      # lines

# Require at least this many distinct Python signals to pass
MIN_PYTHON_SIGNAL_COUNT = 2

PYTHON_SIGNALS   = [
    "def ",
    "class ",
    "import ",
    "return ",
    "self.",
    "print(",
    "if __name__",
    "raise ",
    "yield ",
    "with ",
    "assert ",
    "lambda ",
]

# Patterns that indicate low-quality or incomplete code
NOISE_PATTERNS   = [
    r"^#+\s",                   # markdown headers
    r"^\s*```",                 # markdown fences
    r"^\s*\.\.\.",              # ellipsis-only lines
    r"^(TODO|FIXME|HACK|XXX)",  # unfinished stubs
    r"^pass\s*$",               # bare pass with nothing else
]

# ─── Text Cleaning ────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def fix_encoding(text: str) -> str:
    """Fix common encoding artifacts."""
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\r\n":   "\n",
        "\r":     "\n",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize indentation to 4-space standard.
    Strips trailing whitespace per line, removes excessive blank lines.
    """
    lines   = text.splitlines()
    cleaned = [line.rstrip() for line in lines]

    result  = []
    blanks  = 0
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
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ─── Validation ───────────────────────────────────────────────────────────────

def count_python_signals(code: str) -> int:
    """Count how many distinct Python signals appear in the code."""
    return sum(1 for signal in PYTHON_SIGNALS if signal in code)


def has_docstring(code: str) -> bool:
    """Return True if the code contains a docstring."""
    return '"""' in code or "'''" in code


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


def has_meaningful_body(code: str) -> bool:
    """
    Return True if the code has substance beyond just a signature.
    Rejects functions that are only a signature + pass/return None.
    """
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    body_lines = [
        l for l in lines
        if not l.startswith("def ")
        and not l.startswith("@")
        and not l.startswith('"""')
        and not l.startswith("'''")
        and l not in ("pass", "...", "return", "return None")
    ]
    return len(body_lines) >= 3


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

    signal_count = count_python_signals(code)
    if signal_count < MIN_PYTHON_SIGNAL_COUNT:
        return False, f"too few Python signals ({signal_count})"

    if not is_parseable(code):
        return False, "AST parse failed (syntax error)"

    if has_noise_lines(code):
        return False, "contains markdown or stub noise"

    if not has_meaningful_body(code):
        return False, "body too trivial (stub or pass-only)"

    if len(instruction.strip()) < 15:
        return False, "instruction too short"

    return True, ""

# ─── Quality Scoring ─────────────────────────────────────────────────────────

def compute_quality_score(sample: dict) -> float:
    """
    Compute a quality score for a sample.
    Higher is better. Used by formatter to sort and prioritise.
    """
    code  = sample.get("output", "")
    lines = code.splitlines()

    score = 0.0
    score += min(len(lines) / 15, 3.0)          # length reward (capped)
    score += code.count("def ") * 0.5            # function definitions
    score += code.count("#") * 0.15              # inline comments
    score += 0.5 if has_docstring(code) else 0   # docstring bonus
    score += count_python_signals(code) * 0.1    # Python signal density
    score -= code.count("...") * 0.5             # ellipsis = incomplete
    score -= code.count("TODO") * 0.4
    score -= code.count("pass") * 0.3            # bare pass = stub

    return round(score, 2)

# ─── Batch Cleaning ───────────────────────────────────────────────────────────

def clean_dataset(samples: list[dict]) -> list[dict]:
    """
    Run the full clean + validate pipeline over a list of samples.
    Attaches a quality score to each sample's metadata.
    Returns only valid, cleaned samples.
    """
    cleaned  = []
    rejected = 0

    for sample in samples:
        sample["output"]      = clean_code(sample.get("output", ""))
        sample["instruction"] = clean_instruction(sample.get("instruction", ""))

        valid, reason = is_valid_sample(sample)

        if valid:
            if "metadata" not in sample:
                sample["metadata"] = {}
            sample["metadata"]["code_score"] = compute_quality_score(sample)
            cleaned.append(sample)
        else:
            rejected += 1
            logger.debug(f"Rejected: {reason} | {sample.get('instruction', '')[:60]}")

    logger.info(f"Cleaning complete: {len(cleaned)} kept, {rejected} rejected")
    return cleaned