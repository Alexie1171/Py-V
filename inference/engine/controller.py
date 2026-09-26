"""
controller.py — PY-V (inference/engine/)
Which mode a message needs — chat, explain, generate (write code), debug (fix),
refactor (improve) — by word rules, instantly. Each rule is a pattern with a
weight; a mode's score is the sum of the weights of its rules that match.

A message the rules can't place clearly gets the flag "unclear", and ChatEngine
lets the brain decide when config intent.brain_for_unclear is on (off since the
2026-09-26 laptop check — see intent_classifier.py):
  - no rule matched, but it is pasted code (→ explain) or mentions something
    technical (→ generate, explain if it is a question)
  - the two best modes are nearly tied (within TIE_MARGIN) — the best one stands
Otherwise the best mode wins; nothing matched and nothing technical → chat.
Messages it must get right: experiments/intent_cases.json (python -m experiments.eval_intent).
"""

import re
from dataclasses import dataclass
from typing import List

from inference.engine.language_detector import detect_language


@dataclass
class IntentResult:
    mode: str
    confidence: float
    flags: List[str]


_I = re.IGNORECASE

# (mode, weight, pattern) — every matching pattern adds its weight once
RULES = [
    ("debug", 1.5, re.compile(r"\b[A-Z]\w*(Error|Exception)\b")),   # a real error name (case-sensitive)
    ("debug", 1.0, re.compile(r"\b(errors?|exceptions?|traceback|stack ?trace|bugs?|buggy|broken|"
                              r"crash(es|ed|ing)?|fail(s|ed|ing)?|fix(es|ed|ing)?|raises?)\b", _I)),
    ("debug", 1.0, re.compile(r"not working|doesn'?t work|does not work|isn'?t working|won'?t (run|work)|"
                              r"never stops|infinite loop|wrong (output|result|answer|value)|output is wrong|"
                              r"instead of", _I)),

    ("refactor", 0.9, re.compile(r"\b(refactor(ing)?|improve|optimi[sz]e|clean(er|\s*up)|simplif(y|ied)|"
                                 r"tidy|rewrite)\b", _I)),
    ("refactor", 0.9, re.compile(r"pythonic|readab(le|ility)|\bfaster\b|more efficient|\befficiency\b|"
                                 r"better way|best practices?|\bshorter\b|less code", _I)),

    ("generate", 0.8, re.compile(r"\b(write|create|generate|build|implement)\b", _I)),
    ("generate", 0.8, re.compile(r"\b(give|show) me (a |an |some )?(code|function|script|class|program|"
                                 r"example|snippet)", _I)),
    ("generate", 0.8, re.compile(r"\b(function|script|program|class|method|snippet|code|regex|query)\s+"
                                 r"(that|which|to|for)\b", _I)),
    ("generate", 0.8, re.compile(r"\bmake (a|an|me a|me an) \w+", _I)),
    # "give an example on type script" — without "me" it matched nothing and came out empty in chat
    ("generate", 0.8, re.compile(r"\b(give|show|write)( me)? (an? |some )?(quick |short |simple |small )?examples?\b|"
                                 r"\bexamples? (of|on|in|for|with)\b|\bhello,? world\b", _I)),
    ("generate", 0.8, re.compile(r"\bhow (do|can|would|should) (i|you|we)\b|\bhow to\b", _I)),

    ("explain", 0.9, re.compile(r"\b(explain|describe|meaning|definition)\b", _I)),
    ("explain", 0.9, re.compile(r"\bwhat (does|do|is|are)\b|\bwhat's\b|\bhow does\b|"
                                r"\bhow do (\w+ )?works?\b", _I)),
    ("explain", 0.9, re.compile(r"\bdifference between\b|\bwhy (is|are|does|do|would|should)\b|"
                                r"\btell me about\b|\bwhat happens\b", _I)),

    ("chat", 1.0, re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening|night))\b(?!,? world)", _I)),
    ("chat", 1.0, re.compile(r"\b(remember|my name|who are you|how are you|last (week|time))\b|"
                             r"\b(what|did) (did )?(we|i) (decide|decided|chose|say|said|talk|talked)\b", _I)),
    # Questions about V itself — beats explain's "what is / what are" ("what is your name?")
    ("chat", 1.5, re.compile(r"\byour (name|capabilities|abilities|skills|features|purpose|job|creator|maker|"
                             r"favou?rite)\b|\bwhat can you do\b|\bwhat are you\b|\babout yourself\b|"
                             r"\bwho (made|created|built|trained|owns) you\b|\bhow (do|does) (you|v) work\b|"
                             r"\bare you (an? )?(ai|bot|robot|human|real|person|machine|model|language model|llm)\b", _I)),
    # Questions about the user's computer — V sees its numbers in chat mode (machine.py)
    ("chat", 1.2, re.compile(r"\b(my|this|the) (laptop|computer|pc|machine|gpu|graphics card|ram|cpu|processor)\b|"
                             r"\b(ram|memory|gpu|cpu) (usage|load)\b", _I)),
]

# Pasted code: a fenced block or a line that starts like Python
CODE = re.compile(r"```|^\s*(def|class)\s+\w+|^\s*(import\s+\w+|from\s+\w+\s+import)\b|^\s*return\b|"
                  r"^\s*(for|while|if|elif)\b.*:\s*$", re.MULTILINE)
# Something technical in a message no rule matched ("sort a list of tuples in python");
# any programming language language_detector.py recognises counts too ("fibonacci in haskell")
TECH = re.compile(r"\b(python|code|lists?|dicts?|dictionar(y|ies)|tuples?|sets?|strings?|arrays?|loops?|"
                  r"functions?|class(es)?|files?|csv|json|regex|sql|api|variables?|modules?|import|numpy|"
                  r"pandas|sort(ed|ing)?|recursion|decorators?|generators?|scripts?)\b", _I)
TIE_MARGIN = 0.5   # best − second-best below this → unclear


class Controller:

    def detect_mode(self, user_input: str) -> IntentResult:
        scores = {}
        for mode, weight, pattern in RULES:
            if pattern.search(user_input):
                scores[mode] = scores.get(mode, 0.0) + weight
        has_code = bool(CODE.search(user_input))

        if not scores:
            if has_code:
                return IntentResult(mode="explain", confidence=0.3, flags=["unclear", "code_without_request"])
            if TECH.search(user_input) or detect_language(user_input):   # wants code; a question wants words
                mode = "explain" if "?" in user_input else "generate"
                return IntentResult(mode=mode, confidence=0.3, flags=["unclear", "no_rule_technical"])
            return IntentResult(mode="chat", confidence=0.4, flags=["fallback"])

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)   # ties keep RULES order
        (best, best_score), second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0.0))
        flags = ["code"] if has_code else []
        if second[1] and best_score - second[1] < TIE_MARGIN:
            flags += ["unclear", f"near_tie_{second[0]}"]

        return IntentResult(mode=best, confidence=round(min(0.99, best_score / 3.0), 2), flags=flags)
