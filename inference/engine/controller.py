# inference/engine/controller.py

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class IntentResult:
    mode: str
    confidence: float
    flags: List[str]


class Controller:

    def __init__(self):
        # weighted intent signals
        self.rules = {
            "debug": {
                "keywords": ["error", "exception", "traceback", "bug", "fix", "broken"],
                "weight": 1.0
            },
            "explain": {
                "keywords": ["explain", "what does", "how does", "meaning"],
                "weight": 0.9
            },
            "refactor": {
                "keywords": ["refactor", "improve", "optimize", "clean"],
                "weight": 0.9
            },
            "generate": {
                "keywords": ["write", "create", "generate", "build", "implement"],
                "weight": 0.8
            },
        }

    def _score_mode(self, text: str, mode: str, rule: Dict) -> float:
        score = 0.0
        for kw in rule["keywords"]:
            if re.search(rf"\b{kw}\b", text):
                score += 1.0
        return score * rule["weight"]

    def detect_mode(self, user_input: str) -> IntentResult:
        text = user_input.lower()

        scores: List[Tuple[str, float]] = []

        for mode, rule in self.rules.items():
            score = self._score_mode(text, mode, rule)
            scores.append((mode, score))

        # default fallback
        best_mode, best_score = max(scores, key=lambda x: x[1])

        flags = []

        # edge case detection
        if "code" in text and best_mode == "chat":
            best_mode = "generate"
            flags.append("code_hint_override")

        if best_score == 0:
            return IntentResult(mode="chat", confidence=0.4, flags=["fallback"])

        confidence = min(0.99, best_score / 3.0)

        return IntentResult(
            mode=best_mode,
            confidence=round(confidence, 2),
            flags=flags
        )