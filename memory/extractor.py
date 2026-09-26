"""
extractor.py — PY-V (memory/)
Rule-based fact tagging for V's memory — no model calls (a 3B brain is not
reliable enough to judge what matters; revisit with Granite's chat version).
Only the user's own messages are read: V's answers can contain made-up facts.

extract_facts(text) → [(key, fact)]. The key decides what "the same fact" is:
a newer fact with the same key replaces the older one (see store.add_fact).
Statement rules skip questions ("what is my name?" is not a fact).
"""

import re

_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def _slug(text: str, words: int) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:words]) or "x"


def _clean(text: str) -> str:
    return text.strip().strip(" .,:;\"'")


def _sentence(fact: str) -> str:
    fact = fact.strip()
    return fact[0].upper() + fact[1:] + ("" if fact.endswith((".", "!", "?")) else ".")


# ─── statement rules: (pattern, key builder, fact builder) ────────────────────

_VERB_3RD = {"prefer": "prefers", "like": "likes", "love": "loves", "hate": "hates",
             "always use": "always uses", "usually use": "usually uses", "never use": "never uses",
             "want you to": "wants V to", "would like you to": "would like V to",
             "decided to": "decided to", "decided on": "decided on", "chose": "chose",
             "switched to": "switched to", "will use": "will use", "am using": "is using"}

_STATEMENTS = [
    # explicit request to remember
    (re.compile(r"\b(?:please\s+)?(?:remember|note|don'?t forget)(?:\s+that)?[:,]?\s+(?P<x>.{3,200})", re.I),
     lambda m: f"note:{_slug(m['x'], 4)}",
     lambda m: _sentence(_clean(m["x"]))),
    # name
    (re.compile(r"\b(?i:my name is) (?P<x>[A-Z][\w'-]{1,30})"),   # any case for "my name is", the name capitalised
     lambda m: "user:name",
     lambda m: f"The user's name is {m['x']}."),
    # "my laptop has ...", "my project is ...", "my editor is ..."
    (re.compile(r"\bmy (?P<n>laptop|computer|pc|gpu|graphics card|os|editor|ide|project|job|team|company|"
                r"language|main language|stack) (?P<v>is|has|uses|runs) (?P<x>.{2,120})", re.I),
     lambda m: f"user:{_slug(m['n'], 3)}",
     lambda m: _sentence(f"the user's {m['n'].lower()} {m['v'].lower()} {_clean(m['x'])}")),
    # preferences and habits
    (re.compile(r"\bI (?P<v>prefer|like|love|hate|always use|usually use|never use|want you to|would like you to) "
                r"(?P<x>.{2,120})", re.I),
     lambda m: f"preference:{_slug(m['x'], 3)}",
     lambda m: _sentence(f"the user {_VERB_3RD[m['v'].lower()]} {_clean(m['x'])}")),
    # decisions
    (re.compile(r"\b(?:we|I) (?P<v>decided to|decided on|chose|switched to|will use|am using) (?P<x>.{2,120})", re.I),
     lambda m: f"decision:{_slug(m['x'], 3)}",
     lambda m: _sentence(f"the user {_VERB_3RD[m['v'].lower()]} {_clean(m['x'])}")),
]

# ─── mention rules (questions count too): errors, versions, files ─────────────

# the message must be on the same line (":[ \t]*" — "\s*" once swallowed the line break and stored
# "ZeroDivisionError: ```python" as a fact)
_ERROR   = re.compile(r"\b(?P<t>[A-Z][A-Za-z]*(?:Error|Exception))\b(?::[ \t]*(?P<msg>[^\n]{1,120}))?")
_VERSION = re.compile(r"\b(?P<n>python|node|java|typescript|torch|pytorch|transformers|peft|django|flask|fastapi|"
                      r"react|numpy|pandas|cuda)\s*v?(?P<v>\d+(?:\.\d+){1,2})\b", re.I)
_FILE    = re.compile(r"(?<![\w/\\.])(?P<p>[\w-]+(?:[/\\][\w.-]+)*\.(?:py|ipynb|js|ts|tsx|json|ya?ml|md|toml|cfg|ini|txt))\b")
MAX_FILES = 3


def extract_facts(text: str) -> list:
    """[(key, fact sentence)] found in one user message — de-duplicated by key."""
    facts = {}

    for sentence in _SENTENCE.findall(text):
        if sentence.strip().endswith("?"):
            continue
        for pattern, key, fact in _STATEMENTS:
            match = pattern.search(sentence)
            if match:
                facts[key(match)] = fact(match)

    for match in _ERROR.finditer(text):
        detail = _clean(match["msg"] or "").strip("`")
        msg = f": {detail}" if detail and "```" not in (match["msg"] or "") else ""
        facts[f"error:{match['t'].lower()}"] = f"The user ran into {match['t']}{msg}."
    for match in _VERSION.finditer(text):
        name = {"pytorch": "torch"}.get(match["n"].lower(), match["n"].lower())
        facts[f"version:{name}"] = f"The user mentioned {match['n']} {match['v']}."
    for match in list(_FILE.finditer(text))[:MAX_FILES]:
        path = match["p"].replace("\\", "/")
        facts[f"file:{path.lower()}"] = f"The user works with the file {path}."

    return list(facts.items())
