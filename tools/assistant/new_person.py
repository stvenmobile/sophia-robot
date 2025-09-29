# tools/assistant/skills/new_person.py
import os, json, re, pathlib
from typing import Tuple, Dict, Any

PEOPLE_PATH = os.path.expanduser("~/.sophia/people.json")
pathlib.Path(os.path.dirname(PEOPLE_PATH)).mkdir(parents=True, exist_ok=True)

def load_people() -> Dict[str, Any]:
    if not os.path.exists(PEOPLE_PATH):
        return {"people": []}
    try:
        with open(PEOPLE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"people": []}

def save_people(db: Dict[str, Any]) -> None:
    tmp = PEOPLE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, PEOPLE_PATH)

# Name capture
NAME_RX = re.compile(r"\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+)\b")
BARE_NAME_RX = re.compile(r"^\s*([A-Z][a-z]+)\.?\s*$")

QUESTION_LIKE_RX = re.compile(
    r"\?|^\s*(what|who|when|where|why|how|can you|could you|would you|please|tell me|explain)\b",
    re.I,
)

YES_RX = re.compile(r"\b(yes|yeah|yep|that'?s right|correct|right)\b", re.I)
NO_RX  = re.compile(r"\b(no|nope|not|wrong)\b", re.I)

def _clean(s: str) -> str:
    return (s or "").strip().rstrip(".,!?")

class NewPersonSkill:
    """
    Goal for first session: capture CONFIRMED name, then hand control back.
    If the child asks a question at any time, abandon fact-finding and let
    the main assistant answer it immediately (without losing the question).
    """
    def __init__(self):
        self.state = "ask_name"
        self.person: Dict[str, Any] = {}
        self.retry = 0

    # --- helpers
    def extract_name(self, text: str):
        t = (text or "").strip()
        m = NAME_RX.search(t)
        if m:
            return _clean(m.group(1)).capitalize()
        m2 = BARE_NAME_RX.match(t)
        if m2:
            return _clean(m2.group(1)).capitalize()
        return None

    def match(self, text: str, ctx) -> int:
        """Return >0 if this looks like greeting/name-ish."""
        t = (text or "").strip()
        if self.extract_name(t):
            return 2
        if re.search(r"\b(hi|hello|hey)\b", t, re.I):
            return 1
        return 0

    # --- dialog
    def start(self, text: str, ctx) -> str:
        nm = self.extract_name(text or "")
        if nm:
            self.person["name"] = nm
            self.state = "confirm_name"
            return f"So your name is {nm} — is that right?"
        self.state = "ask_name"
        return "Hi, I'm Sophia. What's your name?"

    def step(self, text: str, ctx) -> Tuple[str, bool]:
        t = (text or "").strip()

        # If child asks a question at any time → handoff to main
        if QUESTION_LIKE_RX.search(t) and self.state != "done":
            ctx["handoff_text"] = t
            return "Sure—let’s talk about that!", True

        if self.state == "ask_name":
            nm = self.extract_name(t)
            if nm:
                self.person["name"] = nm
                self.state = "confirm_name"
                self.retry = 0
                return f"So your name is {nm} — is that right?", False
            if self.retry == 0:
                self.retry = 1
                return "I didn’t catch your name. Please say, “My name is …”.", False
            return "That's okay—tell me your name when you’re ready.", False

        if self.state == "confirm_name":
            if YES_RX.search(t):
                nm = self.person.get("name","friend")
                # save profile minimal
                db = ctx["people"]
                found = next((p for p in db["people"] if p.get("name","").lower()==nm.lower()), None)
                if found: found.update(self.person)
                else: db["people"].append({"name": nm})
                from .new_person import save_people as sp
                sp(db)
                ctx["current_person"] = next((p for p in db["people"] if p.get("name","").lower()==nm.lower()), {"name": nm})
                self.state = "done"
                return f"Nice to meet you, {nm}!", True
            if NO_RX.search(t):
                self.person.pop("name", None)
                self.state = "ask_name"
                return "Oops—okay. What’s your name?", False
            # ambiguous → nudge once
            if self.retry == 0:
                self.retry = 1
                nm = self.person.get("name","friend")
                return f"Did I get it right—{nm}?", False
            self.state = "ask_name"
            return "Please tell me your name again.", False

        return "Okay!", True
