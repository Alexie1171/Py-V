"""
github_scraper.py — PY-V Data Pipeline
Upgraded v4: Raised min function length, increased max_functions_per_repo
             for larger repos, tightened docstring quality check.
"""

import os
import ast
import time
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_URL  = "https://raw.githubusercontent.com"
API_BASE = "https://api.github.com"
BRANCHES = ["main", "master", "develop"]

REPO_BLACKLIST = {
    "tensorflow/models",
    "keras-team/keras",
    "home-assistant/core",
    "pytorch/pytorch",
    "langchain-ai/langchain",
    "fighting41love/funNLP",
    "EbookFoundation/free-programming-books",
    "vinta/awesome-python",
    "521xueweihan/HelloGitHub",
}

# Raised from 50 — pre-filters stubs before they reach the cleaner
MIN_FUNCTION_LENGTH = 100


def get_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN", "")
    return {"Authorization": f"token {token}"} if token else {}


def check_rate_limit():
    r = requests.get(f"{API_BASE}/rate_limit", headers=get_headers())
    if r.status_code != 200:
        return
    remaining = r.json()["rate"]["remaining"]
    reset_at  = r.json()["rate"]["reset"]
    logger.info(f"GitHub API calls remaining: {remaining}")
    if remaining < 20:
        wait = max(reset_at - time.time(), 0) + 5
        logger.warning(f"Rate limit low. Sleeping {int(wait)}s...")
        time.sleep(wait)


def get_repo_tree(owner: str, repo: str) -> tuple:
    for branch in BRANCHES:
        url = f"{API_BASE}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        r   = requests.get(url, headers=get_headers(), timeout=15)
        if r.status_code == 200:
            logger.info(f"Using branch '{branch}' for {owner}/{repo}")
            return r.json(), branch
        logger.debug(f"Branch '{branch}' not found for {owner}/{repo}")
    logger.warning(f"No valid branch found for {owner}/{repo}")
    return None, None


def extract_python_paths(tree: dict) -> list:
    return [
        item["path"]
        for item in tree.get("tree", [])
        if item["path"].endswith(".py") and item["type"] == "blob"
    ]


def get_file_content(owner: str, repo: str, branch: str, file_path: str) -> str | None:
    url = f"{RAW_URL}/{owner}/{repo}/{branch}/{file_path}"
    r   = requests.get(url, headers=get_headers(), timeout=10)
    if r.status_code == 200:
        return r.text
    logger.warning(f"Failed to fetch {file_path} ({r.status_code})")
    return None


def extract_functions_from_source(source: str) -> list:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    functions = []
    lines     = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Skip trivial bodies — must have at least 3 statements
        if len(node.body) < 3 and not ast.get_docstring(node):
            continue

        start       = node.lineno - 1
        end         = node.end_lineno
        func_source = "\n".join(lines[start:end])
        args        = [arg.arg for arg in node.args.args]

        # Pre-filter short functions before they reach the cleaner
        if len(func_source.strip()) < MIN_FUNCTION_LENGTH:
            continue

        functions.append({
            "name":      node.name,
            "docstring": ast.get_docstring(node) or "",
            "args":      args,
            "source":    func_source,
        })

    return functions


def is_good_docstring(text: str) -> bool:
    t = text.strip()
    if len(t) < 20:          # raised from 15
        return False
    if t.startswith(":"):
        return False
    if t.startswith(".."):
        return False
    if t.lower().startswith("todo"):
        return False
    return True


def build_samples_from_functions(
    functions: list,
    file_path: str,
    owner:     str,
    repo:      str,
) -> list:
    samples = []

    for fn in functions:
        if fn["docstring"]:
            first_line = fn["docstring"].strip().splitlines()[0].strip()
            if is_good_docstring(first_line):
                instruction = first_line
            else:
                arg_str     = ", ".join(fn["args"]) if fn["args"] else "no arguments"
                instruction = (
                    f"Write a Python function named `{fn['name']}` "
                    f"that takes {arg_str}."
                )
        else:
            arg_str     = ", ".join(fn["args"]) if fn["args"] else "no arguments"
            instruction = (
                f"Write a Python function named `{fn['name']}` "
                f"that takes {arg_str}."
            )

        samples.append({
            "instruction": instruction,
            "output":      fn["source"],
            "metadata": {
                "source":   "github",
                "repo":     f"{owner}/{repo}",
                "file":     file_path,
                "function": fn["name"],
            }
        })

    return samples


def search_python_repos(
    min_stars: int = 300,
    per_page:  int = 10,
    page:      int = 1,
) -> list:
    url    = f"{API_BASE}/search/repositories"
    params = {
        "q":        f"language:python stars:>{min_stars}",
        "sort":     "stars",
        "order":    "desc",
        "per_page": per_page,
        "page":     page,
    }
    r = requests.get(url, headers=get_headers(), params=params, timeout=15)
    if r.status_code != 200:
        logger.error(f"Repo search failed: {r.status_code} {r.text[:200]}")
        return []

    items = r.json().get("items", [])
    return [(item["owner"]["login"], item["name"]) for item in items]


def fetch_repo_samples(
    owner:         str,
    repo:          str,
    max_files:     int = 30,    # was 20
    max_functions: int = 100,   # was 50 — gets more from large repos
) -> list:
    check_rate_limit()

    tree, branch = get_repo_tree(owner, repo)
    if not tree:
        return []

    py_files = extract_python_paths(tree)
    logger.info(f"{owner}/{repo}: {len(py_files)} Python files found")

    all_samples = []

    for file_path in py_files[:max_files]:
        try:
            source = get_file_content(owner, repo, branch, file_path)
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {file_path}, skipping")
            continue

        if not source:
            continue

        functions = extract_functions_from_source(source)
        samples   = build_samples_from_functions(functions, file_path, owner, repo)
        all_samples.extend(samples)

        if len(all_samples) >= max_functions:
            break

        time.sleep(0.3)

    logger.info(f"{owner}/{repo}: {len(all_samples)} samples extracted")
    return all_samples[:max_functions]


def fetch_multiple_repos(
    repos:                  list | None = None,
    use_discovery:          bool = True,
    discovery_stars:        int  = 300,
    discovery_pages:        int  = 10,
    max_files_per_repo:     int  = 30,
    max_functions_per_repo: int  = 100,
) -> list:
    if repos is None:
        repos = []

    if use_discovery:
        for page in range(1, discovery_pages + 1):
            discovered = search_python_repos(
                min_stars=discovery_stars,
                per_page=10,
                page=page,
            )
            repos.extend(discovered)
            logger.info(f"Discovered {len(discovered)} repos on page {page}")

    seen  = set()
    clean = []
    for owner, repo in repos:
        key = f"{owner}/{repo}"
        if key not in seen and key not in REPO_BLACKLIST:
            seen.add(key)
            clean.append((owner, repo))
        elif key in REPO_BLACKLIST:
            logger.info(f"Skipping blacklisted repo: {key}")
    repos = clean

    all_samples = []

    for owner, repo in repos:
        logger.info(f"Scraping {owner}/{repo}...")
        try:
            samples = fetch_repo_samples(
                owner,
                repo,
                max_files=max_files_per_repo,
                max_functions=max_functions_per_repo,
            )
            all_samples.extend(samples)
        except Exception as e:
            logger.error(f"Failed scraping {owner}/{repo}: {e}, skipping")
            continue

    logger.info(f"Total samples collected: {len(all_samples)}")
    return all_samples


if __name__ == "__main__":
    manual_repos = [
        ("psf",          "requests"),
        ("pallets",      "flask"),
        ("tiangolo",     "fastapi"),
        ("scrapy",       "scrapy"),
        ("sqlalchemy",   "sqlalchemy"),
        ("encode",       "httpx"),
        ("pytest-dev",   "pytest"),
        ("numpy",        "numpy"),
        ("pandas-dev",   "pandas"),
        ("scikit-learn", "scikit-learn"),
        ("aio-libs",     "aiohttp"),
        ("django",       "django"),
        ("celery",       "celery"),
        ("pydantic",     "pydantic"),
        ("python-attrs", "attrs"),
        ("paramiko",     "paramiko"),
    ]

    dataset = fetch_multiple_repos(
        repos=manual_repos,
        use_discovery=True,
        discovery_stars=300,
        discovery_pages=10,
        max_files_per_repo=30,
        max_functions_per_repo=100,
    )

    print(f"\nTotal samples: {len(dataset)}")
    for sample in dataset[:2]:
        print("\n--- SAMPLE ---")
        print("INSTRUCTION:", sample["instruction"])
        print("OUTPUT:\n",    sample["output"][:300])
        print("META:",        sample["metadata"])