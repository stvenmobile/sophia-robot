# tools/assistant/skills/new_person.py
import os, json, re

# Storage path (~/.sophia/people.json)
STORE_PATH = os.path.expanduser("~/.sophia/people.json")

# Default prompt style: simple | simple_plus_grownup | pronoun_only
GENDER_PROMPT_STYLE = os.getenv("GENDER_PROMPT_STYLE", "simple_plus_grownup").lower()

# -------- Simple extractors (kid-friendly, forgiving) --------
NAME_RX    = re.compile(r"\b(i[' ]?m|i am|my name is)\s+([A-Z][a-z]+)\b", re.I)
AGE_RX     = re.compile(r"\b(\d{1,2})\b")
CITY_RX    = re.compile(r"\b(in|from|live in)\s+([A-Za-z][A-Za-z\s\-']{2,})\b", re.I)
GENDER_RX  = re.compile(r"\b(girl|boy|grown[- ]?up|adult)\b", re.I)
PRONOUN_RX = re.compile(r"\b(he|she|they)\b", re.I)
GREETING_RX= re.compile(r"\b(hi|hello|hey)\s+sophia\b", re.I)

def guess_name(text):
    m = NAME_RX.search(text or "")
    return m.group(2).title() if m else None

def guess_age(text):
    m = AGE_RX.search(text or "")
    if not m: return None
    try:
        n = int(m.group(1))
        return n if 1 <= n <= 120 else None
    except Exception:
        return None

def guess_city(text):
    m = CITY_RX.search(text or "")
    return m.group(2).strip().title() if m else None

def guess_gender(text):
    m = GENDER_RX.search(text or "")
    if not m: return None
    g = m.group(1).lower()
    if g.startswith("girl"): return "girl"
    if g.startswith("boy"): return "boy"
    if "grown" in g or "adult" in g: return "grown-up"
    return None

def guess_pronoun(text):
    m = PRONOUN_RX.search(text or "")
    if not m: return None
    p = m.group(1).lower()
    return "they" if p == "they" else ("he" if p == "he" else "she")

# -------- tiny JSON store (in this file to keep setup simple) --------
def load_people():
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"people": []}

def save_people(db):
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def find_by_name(db, name):
    if not name: return None
    n = name.strip().lower()
    for p in db["people"]:
        if (p.get("name","").lower() == n):
            return p
    return None

# -------- Skill class --------
class NewPersonSkill:
    name = "new_person"

    def load_people(self):  # exposed for assistant.py
        return load_people()

    def match(self, text, ctx) -> int:
        """
        Score >0 to begin intro flow:
        - 'Hi/Hello/Hey Sophia'
        - A line that looks like self-intro (contains a name pattern)
        - Or 'I'm new'
        """
        if not text: return 0
        t = text.strip()
        if GREETING_RX.search(t): return 10
        if guess_name(t): return 8
        if "i'm new" in t.lower() or "i am new" in t.lower(): return 7
        return 0

    def start(self, text, ctx):
        # Initialize state
        ctx["skill_state"] = {"step": 0, "retries": {"name":0,"gender":0,"pronoun":0,"age":0,"city":0},
                              "person": {"name": None, "gender": None, "pronoun": None, "age": None, "city": None}}
        # If they already said their name in the trigger, capture it and move to gender question
        nm = guess_name(text)
        if nm:
            ctx["skill_state"]["person"]["name"] = nm
            ctx["skill_state"]["step"] = 1
            return self._prompt_gender(nm)
        return "Hi! I’m Sophia. What’s your name?"

    # ----- step machine -----
    def step(self, text, ctx):
        st = ctx["skill_state"]; p = st["person"]; r = st["retries"]
        t = (text or "").strip()

        # STEP 0: get name
        if st["step"] == 0:
            nm = guess_name(t)
            if not nm:
                if r["name"] == 0:
                    r["name"] += 1
                    return "Sorry, I didn’t catch your name. Can you tell me your name?", False
                # fallback
                p["name"] = "Friend"
            else:
                p["name"] = nm
            st["step"] = 1
            return self._prompt_gender(p["name"])

        # STEP 1: gender/pronoun (with one retry)
        if st["step"] == 1:
            style = GENDER_PROMPT_STYLE
            if style == "pronoun_only":
                pr = guess_pronoun(t)
                if not pr:
                    if r["pronoun"] == 0:
                        r["pronoun"] += 1
                        return f"Should I say ‘he’ or ‘she’ when I talk about you?", False
                    p["pronoun"] = "they"
                else:
                    p["pronoun"] = pr
                    p["gender"] = {"he":"boy","she":"girl"}.get(pr, None)
                st["step"] = 2
                return "How old are you?", False
            else:
                g = guess_gender(t)
                if not g:
                    if r["gender"] == 0:
                        r["gender"] += 1
                        if style == "simple_plus_grownup":
                            return "Are you a boy or a girl? If you’re an adult you can say ‘grown-up’.", False
                        else:
                            return "Are you a boy or a girl?", False
                    p["gender"] = "unspecified"
                else:
                    p["gender"] = g
                    if g == "grown-up":
                        st["step"] = "ask_pronoun"
                        return "Got it! Should I say ‘he’ or ‘she’ when I talk about you?", False
                st["step"] = 2
                return "How old are you?", False

        # Optional pronoun step for grown-ups
        if st["step"] == "ask_pronoun":
            pr = guess_pronoun(t)
            if not pr:
                if r["pronoun"] == 0:
                    r["pronoun"] += 1
                    return "Should I say ‘he’ or ‘she’ when I talk about you?", False
                p["pronoun"] = "they"
            else:
                p["pronoun"] = pr
            st["step"] = 2
            return "How old are you?", False

        # STEP 2: age (retry once)
        if st["step"] == 2:
            ag = guess_age(t)
            if ag is None:
                if r["age"] == 0:
                    r["age"] += 1
                    return "How old are you? You can just say a number.", False
                p["age"] = "unspecified"
            else:
                p["age"] = ag
            st["step"] = 3
            return "And what city do you live in?", False

        # STEP 3: city (retry once) → save
        if st["step"] == 3:
            city = guess_city(t)
            if not city:
                if r["city"] == 0:
                    r["city"] += 1
                    return "What city do you live in?", False
                p["city"] = "Unspecified"
            else:
                p["city"] = city

            # SAVE / UPSERT
            db = ctx["people"]
            existing = find_by_name(db, p["name"])
            if existing:
                existing.update({k:v for k,v in p.items() if v})
            else:
                db["people"].append(p)
            save_people(db)

            ctx["skill_state"] = {}
            who = p["name"] or "friend"
            return f"Awesome! I’ll remember you, {who}. Do you want to ask me something now?", True

        # Fallback safety
        ctx["skill_state"] = {}
        return "Let’s chat! What would you like to know?", True

    # ----- helpers -----
    def _prompt_gender(self, name):
        style = GENDER_PROMPT_STYLE
        if style == "pronoun_only":
            return f"Nice to meet you, {name}! Should I say ‘he’ or ‘she’ when I talk about you?"
        if style == "simple_plus_grownup":
            return f"Nice to meet you, {name}! Are you a boy or a girl… or a grown-up?"
        return f"Nice to meet you, {name}! Are you a boy or a girl?"
