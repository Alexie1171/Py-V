"""
stackoverflow_scraper.py — PY-V Data Pipeline
Upgraded v4: Raised minimum code score to match formatter threshold,
             added minimum line count gate, tightened Python signal check.
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

API_BASE = "https://api.stackexchange.com/2.3"
SO_KEY   = ""

PYTHON_SIGNALS = [
    "def ", "import ", "class ", "print(", "self.",
    "return ", "if __name__", "lambda ", "range(",
    "len(", "isinstance(", "dict(", "list(", "tuple(",
    "raise ", "yield ", "with ", "assert ",
]

# Match the formatter's MIN_CODE_SCORE so SO samples face the same bar
MIN_CODE_SCORE   = 1.0
MIN_CODE_LINES   = 5      # mirror cleaner's MIN_LINE_COUNT
MIN_CODE_LENGTH  = 100    # mirror cleaner's MIN_CODE_LENGTH


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def extract_code_blocks(html: str) -> list[str]:
    pre_blocks    = re.findall(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>", html, re.DOTALL)
    inline_blocks = re.findall(r"<code>(.*?)</code>", html, re.DOTALL)
    blocks = pre_blocks if pre_blocks else inline_blocks
    return [unescape(b).strip() for b in blocks]


def is_python_code(code: str) -> bool:
    """Require at least 2 Python signals — stricter than before."""
    hits = sum(1 for signal in PYTHON_SIGNALS if signal in code)
    return hits >= 2


def score_code_block(code: str) -> float:
    """
    Quality score aligned with cleaner.compute_quality_score().
    Higher = better training sample.
    """
    score = 0.0
    lines = code.strip().splitlines()

    score += min(len(lines) / 15, 3.0)
    score += code.count("def ") * 0.5
    score += code.count("#") * 0.15
    score += 0.5 if ('"""' in code or "'''" in code) else 0
    score -= code.count("...") * 0.5
    score -= code.count("TODO") * 0.4
    score -= code.count("pass") * 0.3

    return round(score, 2)


def fetch_accepted_answers(answer_ids: list[int]) -> dict[int, str]:
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


def fetch_questions(
    tag:       str = "python",
    pagesize:  int = 20,
    page:      int = 1,
    sort:      str = "votes",
    min_score: int = 3,
) -> list[dict]:
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


def build_samples_from_question(
    question:      dict,
    answer_bodies: dict[int, str],
    seen_codes:    set,
) -> list[dict]:
    title       = clean_html(question.get("title", ""))
    q_body      = question.get("body", "")
    answer_id   = question.get("accepted_answer_id")
    answer_body = answer_bodies.get(answer_id, "") if answer_id else ""

    if not answer_body:
        return []

    raw_blocks  = extract_code_blocks(answer_body)
    answer_text = clean_html(answer_body)
    samples     = []

    for code in raw_blocks:
        # Length and line count gates — mirrors cleaner thresholds
        if len(code) < MIN_CODE_LENGTH:
            continue
        lines = code.strip().splitlines()
        if len(lines) < MIN_CODE_LINES:
            continue
        if not is_python_code(code):
            continue
        if code in seen_codes:
            continue

        score = score_code_block(code)
        if score < MIN_CODE_SCORE:
            continue

        seen_codes.add(code)

        q_codes = extract_code_blocks(q_body)
        context = ""
        if q_codes:
            best_q_code = max(q_codes, key=len)
            if len(best_q_code) > 30:
                context = f"\n\nContext from question:\n```python\n{best_q_code}\n```"

        instruction = f"{title}{context}"
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


def fetch_stackoverflow_samples(
    tags:        list[str] = None,
    pages:       int = 20,
    pagesize:    int = 20,
    min_q_score: int = 3,
) -> list[dict]:
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

            answered   = [q for q in questions if q.get("accepted_answer_id")]
            answer_ids = [q["accepted_answer_id"] for q in answered]

            answer_bodies = fetch_accepted_answers(answer_ids)

            for question in answered:
                samples = build_samples_from_question(question, answer_bodies, seen_codes)
                all_samples.extend(samples)

            time.sleep(1.0)

    all_samples.sort(key=lambda s: s["metadata"]["code_score"], reverse=True)
    logger.info(f"Total SO samples collected: {len(all_samples)}")
    return all_samples


if __name__ == "__main__":
    dataset = fetch_stackoverflow_samples(
        tags=["python", "python-3.x"],
        pages=20,
        pagesize=20,
        min_q_score=3,
    )

    print(f"\nTotal samples: {len(dataset)}")
    for sample in dataset[:3]:
        print("\n--- SAMPLE ---")
        print("INSTRUCTION:", sample["instruction"][:200])
        print("CODE:\n",      sample["output"][:300])
        print("META:",        sample["metadata"])