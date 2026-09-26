"""
lookup.py — PY-V (learning/)
Looking things up online, after asking (Phase 13, owner's design): when V
doesn't know something she asks "Want me to look that up online?" and
searches only on a yes — in any wording ("yes", "yea", "sure, look it up") —
and not on a no ("no", "nah", "don't"). Asking her to look something up is
itself the yes.

  answer_kind(message)        "yes" / "no" / None (neither — a new message)
  asked_to_look_up(message)   the words to search when the user asks for it
  offer_lookup(question, answer, mode)   True when her answer shows she doesn't know
  build_query(question)       the search words — only these leave the laptop

Rules only, no brain: instant, and a 3B brain is not reliable at judging this.
"""

import re
from typing import Optional

_I = re.IGNORECASE

# Yes in any wording — the start of the message decides ("yes, but only the docs" is a yes)
_YES = re.compile(
    r"^\s*(y|ya|yah|yas|yass|ye|yea|yeah|yeh|yep|yup|yes+|ok(ay)?|k|kk|sure|sure thing|of course|ofc|definitely|"
    r"absolutely|certainly|go( on| ahead| for it)?|do it|go ahead and look|let'?s do it|why not|"
    r"alright|all right|affirmative|aye|yes please|yeah please|sounds good|that would be (great|nice|good)|"
    r"look it up|search( it| for it)?|google it|check( it)?( online)?|find (it|out))\b", _I)
# No in any wording
_NO = re.compile(
    r"^\s*(n|no+|nah+|nope|na|nay|not now|not really|no thanks|no thank you|no need|don'?t|do not|never ?mind|"
    r"nevermind|skip( it)?|leave it|forget it|cancel|stop|not necessary|i'?m good|we'?re good|it'?s fine|that'?s fine|"
    r"no worries|pass)\b", _I)
# "please" alone is a yes; "please explain more" is a new request
_PLEASE = re.compile(r"^\s*(please|pls|plz)(\s+(do( it)?|go ahead|look( it)? up|search( it)?))?\s*[.!]*\s*$", _I)
_NEGATED_YES = re.compile(r"^\s*(no|don'?t|do not)\b.*\b(look|search|google)", _I)

# The user asks her to look something up (the ask itself is the permission)
_ASK_LOOKUP = [
    re.compile(r"\b(?:look|search)\s+(?:up\s+)?(?:online|on the (?:web|internet)|the (?:web|internet))\s+(?:for\s+)?(.+)", _I),
    re.compile(r"\b(?:look up|google|search for|search the (?:web|internet) for|search online for|find online|"
               r"check online(?: for)?)\s+(.+)", _I),
    re.compile(r"\b(?:look|search) (?:it|that|this) up\b()", _I),
    re.compile(r"\bcan you (?:look|search) (?:it|that|this|online)\b()", _I),
]

# Her answer says she doesn't know
_DONT_KNOW = re.compile(
    r"\b(i don'?t know|i do not know|i'?m not sure|i am not sure|i'?m not certain|not (entirely )?sure|"
    r"i (don'?t|do not) have (any |enough |specific |up-to-date |current |real-time )?(information|info|data|details|access|knowledge)|"
    r"i (can'?t|cannot|am unable to|'m unable to) (browse|access|search|look up|check)|"
    r"(as of|up to) my (last )?(knowledge|training)|my (knowledge|training) (cutoff|cut-off|only goes)|"
    r"i'?m not (aware|familiar)|i am not (aware|familiar)|no information (about|on)|i couldn'?t find|"
    r"beyond my knowledge|i have no (idea|information))\b", _I)
# The question needs up-to-date facts
_RECENT = re.compile(
    r"\b(latest|newest|most recent|today'?s|this (week|month|year)|news|release date|been released|"
    r"(current|latest|newest|new) (stable )?(version|release)|20[2-9]\d)\b", _I)

_STOP = set("""
a an the is are was were be been being to of in on at for from by with about as into and or but if then so
what which who whom whose when where why how do does did can could would should will shall may might must
i me my you your we our it its this that these those there here please tell explain show give know let
look search google online web internet up v hey hi thanks thank
""".split())


def answer_kind(message: str) -> Optional[str]:
    """"yes" / "no" to "Want me to look that up online?", None when the message is something else."""
    text = message.strip()
    if not text:
        return None
    if _NEGATED_YES.search(text) or _NO.match(text):
        return "no"
    # a yes is short, or about looking it up — "please explain more" is a new message, not a yes
    if _PLEASE.match(text):
        return "yes"
    if _YES.match(text) and (len(text.split()) <= 6 or re.search(r"\b(look|search|google|online|web)\b", text, _I)):
        return "yes"
    return None


def asked_to_look_up(message: str, previous_question: str = "") -> Optional[str]:
    """The search words when the user asks her to look something up; "look it up" → the previous question."""
    for pattern in _ASK_LOOKUP:
        match = pattern.search(message)
        if match:
            subject = (match.group(1) or "").strip(" ?.!")
            if subject and subject.lower() not in ("it", "that", "this", "online"):
                return build_query(subject)
            if previous_question:
                return build_query(previous_question)
            return None
    return None


def offer_lookup(question: str, answer: str, mode: str) -> bool:
    """Offer a lookup after a chat / explain answer that says she doesn't know, or on a question
    that needs up-to-date facts (her knowledge stops at her training)."""
    if mode not in ("chat", "explain"):
        return False
    if _DONT_KNOW.search(answer):
        return True
    return bool(_RECENT.search(question)) and "?" in question and len(question.split()) >= 4


def build_query(question: str, max_words: int = 12) -> str:
    """Search words from a question: no code, no chat words, at most max_words. Only these go online."""
    text  = re.sub(r"```.*?```", " ", question, flags=re.S)
    text  = re.sub(r"`[^`]*`", " ", text)
    words = re.findall(r"[\w.+#-]+", text)
    kept  = [w for w in words if w.lower() not in _STOP and len(w) > 1]
    return " ".join(kept[:max_words])
