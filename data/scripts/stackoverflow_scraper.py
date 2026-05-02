"""
stackoverflow_scraper.py — PY-V Data Pipeline
Upgraded v3: Accepted answer fetching, HTML cleaning, Python code filtering,
             multi-block extraction, quality scoring, dedup guard.
"""

import re
import time
import logging
import requests
from html import unescape
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

API_BASE = "https://api.stackexchange.com/2.3"
SO_KEY   = ""   # Optional: set your StackExchange API key for higher quota

# Python keywords used to filter genuine Python code blocks
PYTHON_SIGNALS = [
    "def ", "import ", "class ", "print(", "self.",
    "return ", "if __name__", "lambda ", "range(",
    "len(", "isinstance(", "dict(", "list(", "tuple(",
]

# ─── HTML & Code Cleaning ─────────────────────────────────────────────────────

def clean_html(text: str) -> str:
    """Strip all HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def extract_code_blocks(html: str) -> list[str]:
    """
    Extract all <code> blocks. Prefer <pre><code> (multi-line) over
    inline <code> (usually single tokens).
    """
    # Multi-line blocks inside <pre> — highest priority
    pre_blocks = re.findall(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", html, re.DOTALL)

    # Fallback: bare <code> tags
    inline_blocks = re.findall(r"<code>(.*?)</code>", html, re.DOTALL)

    blocks = pre_blocks if pre_blocks else inline_blocks
    return [unescape(b).strip() for b in blocks]


def is_python_code(code: str) -> bool:
    """Heuristic check: does this code block look like Python?"""
    return any(signal in code for signal in PYTHON_SIGNALS)


def score_code_block(code: str) -> float:
    """
    Simple quality score for a code block.
    Higher = better training sample.
    """
    score = 0.0
    lines = code.strip().splitlines()

    score += min(len(lines) / 10, 3.0)          # length reward (up to 3 pts)
    score += code.count("def ") * 0.5            # function definitions
    score += code.count("#") * 0.2               # comments
    score += code.count('"""') * 0.3             # docstrings
    score -= code.count("...") * 0.5             # ellipsis = incomplete code
    score -= code.count("TODO") * 0.3

    return round(score, 2)

# ─── Accepted Answer Fetching ─────────────────────────────────────────────────

def fetch_accepted_answers(answer_ids: list[int]) -> dict[int, str]:
    """
    Batch-fetch answer bodies for a list of answer IDs.
    Returns {answer_id: body_html}.
    """
    if not answer_ids:
        return {}

    ids_str = ";".join(str(i) for i in answer_ids)
    url     = f"{API_BASE}/answers/{ids_str}"
    params  = {
        "site":   "stackoverflow",
        "filter": "withbody",
        "key":    SO_KEY,
    }

    r = requests.get(url, params=params)
    if r.status_code != 200:
        logger.warning(f"Answer fetch failed: {r.status_code}")
        return {}

    return {
        item["answer_id"]: item.get("body", "")
        for item in r.json().get("items", [])
    }

# ─── Question Fetching ────────────────────────────────────────────────────────

def fetch_questions(
    tag:      str = "python",
    pagesize: int = 20,
    page:     int = 1,
    sort:     str = "votes",
    min_score: int = 5,
) -> list[dict]:
    """Fetch high-voted Python questions with bodies."""
    url    = f"{API_BASE}/questions"
    params = {
        "order":    "desc",
        "sort":     sort,
        "tagged":   tag,
        "site":     "stackoverflow",
        "pagesize": pagesize,
        "page":     page,
        "filter":   "withbody",
        "key":      SO_KEY,
    }

    r = requests.get(url, params=params)
    if r.status_code != 200:
        logger.error(f"Question fetch failed: {r.status_code}")
        return []

    items = r.json().get("items", [])
    return [q for q in items if q.get("score", 0) >= min_score]

# ─── Sample Building ──────────────────────────────────────────────────────────

def build_samples_from_question(
    question:       dict,
    answer_bodies:  dict[int, str],
    seen_codes:     set,
) -> list[dict]:
    """
    Build training samples from a question + its accepted answer.
    Returns multiple samples if answer has multiple good code blocks.
    """
    title       = clean_html(question.get("title", ""))
    q_body      = question.get("body", "")
    answer_id   = question.get("accepted_answer_id")
    answer_body = answer_bodies.get(answer_id, "") if answer_id else ""

    if not answer_body:
        return []

    # Extract and score all code blocks from the answer
    raw_blocks = extract_code_blocks(answer_body)
    answer_text = clean_html(answer_body)   # clean prose explanation

    samples = []

    for code in raw_blocks:
        # Quality gates
        if len(code) < 40:
            continue
        if not is_python_code(code):
            continue
        if code in seen_codes:
            continue

        score = score_code_block(code)
        if score < 0.5:
            continue

        seen_codes.add(code)

        # Build the instruction from title + question context codes
        q_codes = extract_code_blocks(q_body)
        context = ""
        if q_codes:
            best_q_code = max(q_codes, key=len)
            if len(best_q_code) > 30:
                context = f"\n\nContext from question:\n```python\n{best_q_code}\n```"

        instruction = f"{title}{context}"

        # Include prose explanation if available
        explanation = answer_text[:500].strip() if answer_text else ""

        samples.append({
            "instruction": instruction,
            "output":      code,
            "explanation": explanation,
            "metadata": {
                "source":      "stackoverflow",
                "question_id": question.get("question_id"),
                "answer_id":   answer_id,
                "q_score":     question.get("score", 0),
                "code_score":  score,
                "tags":        question.get("tags", []),
            }
        })

    return samples

# ─── Main Fetch ───────────────────────────────────────────────────────────────

def fetch_stackoverflow_samples(
    tags:        list[str] = None,
    pages:       int = 3,
    pagesize:    int = 20,
    min_q_score: int = 5,
) -> list[dict]:
    """
    Full pipeline:
    questions → accepted answer IDs → batch fetch answers
    → extract + filter code → build training samples
    """
    if tags is None:
        tags = ["python", "python-3.x"]

    all_samples: list[dict] = []
    seen_codes:  set        = set()

    for tag in tags:
        logger.info(f"Fetching tag: [{tag}]")

        for page in range(1, pages + 1):
            questions = fetch_questions(
                tag=tag,
                pagesize=pagesize,
                page=page,
                min_score=min_q_score,
            )
            logger.info(f"  Page {page}: {len(questions)} questions above score threshold")

            if not questions:
                break

            # Collect only questions with an accepted answer
            answered = [q for q in questions if q.get("accepted_answer_id")]
            answer_ids = [q["accepted_answer_id"] for q in answered]

            answer_bodies = fetch_accepted_answers(answer_ids)

            for question in answered:
                samples = build_samples_from_question(question, answer_bodies, seen_codes)
                all_samples.extend(samples)

            time.sleep(1.0)   # respect API rate limits

    # Sort by code quality score descending
    all_samples.sort(key=lambda s: s["metadata"]["code_score"], reverse=True)

    logger.info(f"Total SO samples collected: {len(all_samples)}")
    return all_samples

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset = fetch_stackoverflow_samples(
        tags=["python", "python-3.x"],
        pages=3,
        pagesize=20,
        min_q_score=10,
    )

    print(f"\nTotal samples: {len(dataset)}")
    for sample in dataset[:3]:
        print("\n--- SAMPLE ---")
        print("INSTRUCTION:", sample["instruction"][:200])
        print("CODE:\n",      sample["output"][:300])
        print("META:",        sample["metadata"])