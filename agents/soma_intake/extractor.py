"""
================================================================================
BLOCK: ACORD-125/126 Field Extractor (confidence + provenance)
--------------------------------------------------------------------------------
Role & contracts
  Input : the case's normalised InboundMessages (any channel, oldest first),
          each split into Sentences with exact character offsets, ASR
          confidence (voice) and conversational context (the question that
          was asked just before).
  Output: CaseExtraction - per field: value, confidence in [0,1], status
          (confident | low | conflict | missing), every supporting Citation
          (message id, channel, char offsets, verbatim quote) and any
          superseded alternatives. Pure function: the same messages always give
          the same output, which the durable workflow's replay relies on.

Key engineering mechanism
  * Deterministic rule engine: 18 extractors (regex + gazetteers) over
    number-normalised text. "one point two million" -> 1.2 million, spoken
    digit runs ("nine four one zero nine") -> 94109, full US state names ->
    USPS codes.
  * Context-aware: a caller saying "About one point two million." right
    after the agent asked about revenue is attributed to annual_revenue
    because the rules see (context + sentence) but take the VALUE only from
    the customer's own sentence.
  * Confidence = rule strength x channel reliability (voice uses the ASR
    segment confidence) x hedge penalty ("about", "I think" -> x0.92).
    Agreeing mentions combine by noisy-OR, 1 - prod(1 - c_i), capped at 0.99.
  * Conflicts: a later mention with a correction cue ("actually", "we just
    hired", "correction") supersedes earlier values. Disagreement WITHOUT a
    cue keeps the best-supported value at confidence x0.5 with
    status=conflict, which forces a confirmation question or a human handoff.
  * Optional LLM pass (real provider only): proposals are accepted only if
    their quoted evidence appears VERBATIM in a source message, so a
    hallucinated value cannot enter the record.

Failure modes & 1-step mitigation
  * Vague or missing answer -> status low/missing; the workflow asks a
    follow-up (max 3) and then hands off to a human.
  * Over-confident rule on noisy ASR -> the ASR confidence multiplies the
    rule confidence, so poor audio pushes the field below the 0.80 gate.
================================================================================
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Iterable

CONFIDENCE_GATE = 0.80

# ------------------------------------------------------------------ schema
@dataclass(frozen=True)
class FieldSpec:
    key: str
    form: str           # ACORD 125 or 126
    label: str          # human label (also used to map portal form fields)
    required: bool
    kind: str = "str"   # str | int | money | date | list | email | phone


FIELDS: list[FieldSpec] = [
    FieldSpec("applicant_name", "125", "Applicant (named insured)", True),
    FieldSpec("entity_type", "125", "Legal entity type", True),
    FieldSpec("fein", "125", "FEIN", False),
    FieldSpec("mailing_street", "125", "Mailing street address", True),
    FieldSpec("mailing_city", "125", "City", True),
    FieldSpec("mailing_state", "125", "State", True),
    FieldSpec("mailing_zip", "125", "ZIP code", True),
    FieldSpec("business_phone", "125", "Business phone", False, "phone"),
    FieldSpec("contact_name", "125", "Contact name", True),
    FieldSpec("contact_email", "125", "Contact email", True, "email"),
    FieldSpec("years_in_business", "125", "Years in business", False, "int"),
    FieldSpec("business_description", "125", "Nature of business / description of operations", True),
    FieldSpec("naics", "125", "NAICS code", False),
    FieldSpec("employees_full_time", "125", "Full-time employees", True, "int"),
    FieldSpec("employees_part_time", "125", "Part-time employees", False, "int"),
    FieldSpec("annual_revenue", "125", "Annual revenue (gross sales)", True, "money"),
    FieldSpec("policy_effective_date", "125", "Proposed effective date", True, "date"),
    FieldSpec("lines_of_business", "125", "Lines of business requested", True, "list"),
    FieldSpec("gl_each_occurrence", "126", "Each occurrence limit", True, "money"),
    FieldSpec("gl_general_aggregate", "126", "General aggregate limit", True, "money"),
    FieldSpec("gl_deductible", "126", "Deductible", False, "money"),
    FieldSpec("prior_losses_3y", "126", "Losses in last 3 years", True, "int"),   # loss history: underwriting needs it
    FieldSpec("prior_carrier", "125", "Prior carrier", True),
    FieldSpec("hazards", "126", "Hazards / exposures", False, "list"),
]
SPEC = {f.key: f for f in FIELDS}
REQUIRED = [f.key for f in FIELDS if f.required]

BUSINESS_TYPES: dict[str, tuple[tuple[str, ...], str, str]] = {
    # key: (keywords, description, NAICS 2017 - illustrative mapping)
    "bakery": (("bakery", "bake shop", "pastry shop", "bakeshop"), "Retail bakery", "311811"),
    "coffee_shop": (("coffee shop", "cafe", "café", "espresso bar"), "Coffee shop / cafe", "722515"),
    "florist": (("florist", "flower shop"), "Retail florist", "453110"),
    "landscaping": (("landscaping", "lawn care", "landscaper"), "Landscaping services", "561730"),
    "it_consulting": (("it consulting", "software consulting", "managed it"), "IT consulting", "541512"),
    "dental_office": (("dental office", "dental practice", "dentist"), "Dental office", "621210"),
    "yoga_studio": (("yoga studio", "pilates studio"), "Yoga / fitness studio", "713940"),
    "cleaning_service": (("cleaning service", "janitorial", "cleaning company"), "Commercial cleaning", "561720"),
    "auto_repair": (("auto repair", "mechanic shop", "car repair"), "Auto repair shop", "811111"),
    "bookstore": (("bookstore", "book shop", "bookshop"), "Retail bookstore", "451211"),
    "food_truck": (("food truck",), "Food truck", "722330"),
    "photography": (("photography studio", "photographer", "photo studio"), "Photography studio", "541921"),
    "bar": (("cocktail bar", "neighborhood bar", "pub", "tavern"), "Bar / tavern", "722410"),
    "roofing": (("roofing", "roofer"), "Roofing contractor", "238160"),
    "daycare": (("daycare", "day care", "child care center", "preschool"), "Child day care", "624410"),
    "fireworks": (("fireworks",), "Fireworks retail", "453998"),
    "brewery": (("brewery", "taproom", "brewpub"), "Brewery taproom", "312120"),
}
HAZARDS: dict[str, tuple[tuple[str, ...], bool]] = {
    # key: (keywords, high_risk)
    "liquor": (("serve beer", "serve wine", "serve alcohol", "liquor", "cocktails", "full bar", "taproom", "pour beer"), True),
    "roof_work": (("roof", "roofing", "on rooftops"), True),
    "childcare": (("children", "kids in our care", "daycare", "toddlers", "infants"), True),
    "explosives": (("fireworks", "explosive", "pyrotechnic"), True),
    "deep_fryer": (("deep fryer", "fryer", "fry donuts", "frying"), False),
    "delivery": (("deliveries", "delivery van", "we deliver", "delivery driver"), False),
    "offsite_catering": (("catering", "cater events", "off-site events"), False),
    "power_tools": (("power tools", "chainsaw", "mowers", "lifts"), False),
}
HIGH_RISK = {k for k, (_, hr) in HAZARDS.items() if hr}
LINES = {
    "general_liability": ("general liability", "liability insurance", "gl coverage", "gl policy", " gl "),
    "property": ("property coverage", "property insurance", "cover our equipment", "building and contents"),
    "workers_comp": ("workers comp", "workers' comp", "workers compensation"),
    "commercial_auto": ("commercial auto", "auto policy for the van", "vehicle coverage"),
}
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
    "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}

# ------------------------------------------------------------------ message model (shared with workflow.py)
@dataclass
class Sentence:
    idx: int
    start: int
    end: int
    text: str
    context: str = ""          # e.g. the agent's question that preceded this reply
    asr_confidence: float = 1.0


@dataclass
class InboundMessage:
    message_id: str
    channel: str               # email | sms | voice
    contact: str               # email address or E.164 phone
    received_at: str           # ISO-8601
    text: str
    sentences: list[Sentence]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InboundMessage":
        return cls(**{**d, "sentences": [Sentence(**s) for s in d["sentences"]]})


# a terminator only ends a sentence when followed by whitespace/end, so "$1.5M" and "e.g." stay intact
SENT_RE = re.compile(r"(?:[^.!?\n]|[.!?](?=\S))+(?:[.!?]+|\n|$)")


def split_sentences(text: str, context: str = "", asr_confidence: float = 1.0, offset: int = 0, start_idx: int = 0) -> list[Sentence]:
    out = []
    for m in SENT_RE.finditer(text):
        s = m.group().strip()
        if s:
            lead = len(m.group()) - len(m.group().lstrip())
            out.append(Sentence(start_idx + len(out), offset + m.start() + lead, offset + m.start() + lead + len(s), s,
                                context, asr_confidence))
    return out


# ------------------------------------------------------------------ number normalisation
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALES = {"hundred": 100, "thousand": 1_000}
_NUMWORD = r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|" \
           r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|and|a)"
_PHRASE_RE = re.compile(rf"\b{_NUMWORD}(?:[\s-]+{_NUMWORD})*\b", re.I)
_DIGITWORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)"
_DIGITRUN_RE = re.compile(rf"\b{_DIGITWORD}(?:[\s,-]+{_DIGITWORD}){{2,}}\b", re.I)
_POINT_RE = re.compile(r"\b(\d+) point (\d)\b", re.I)   # applied AFTER word->digit conversion


def _words_to_int(words: list[str]) -> int | None:
    total, cur, seen = 0, 0, False
    for w in words:
        w = w.lower()
        if w == "and":
            continue
        if w in _UNITS:
            cur += _UNITS[w]; seen = True
        elif w in _TENS:
            cur += _TENS[w]; seen = True
        elif w == "hundred":
            cur = (cur or 1) * 100; seen = True
        elif w == "thousand":
            total += (cur or 1) * 1000; cur = 0; seen = True
        else:
            return None
    return total + cur if seen else None


def normalize_numbers(text: str) -> str:
    """Spoken numbers -> digits. 'twelve' -> 12, 'one point two million' -> 1.2 million, 'a dozen' -> 12."""
    t = re.sub(r"\ba dozen\b", "12", text, flags=re.I)
    t = re.sub(r"\bhalf a million\b", "500 thousand", t, flags=re.I)
    t = _DIGITRUN_RE.sub(lambda m: "".join(str(_UNITS.get(w.lower(), 0)) for w in re.split(r"[\s,-]+", m.group())), t)

    t = re.sub(r"\ba (hundred|thousand)\b", r"one \1", t, flags=re.I)

    def phrase(m: re.Match) -> str:
        # "and" joins only after a scale word ("one hundred and five"); otherwise it separates numbers
        words, segs, cur = re.split(r"[\s-]+", m.group()), [], []
        for i, w in enumerate(words):
            if w.lower() == "and" and not (i and words[i - 1].lower() in ("hundred", "thousand")):
                segs.append(cur); segs.append(["and"]); cur = []
            elif w.lower() == "a":
                segs.append(cur); segs.append(["a"]); cur = []
            else:
                cur.append(w)
        segs.append(cur)
        out = []
        for seg in segs:
            if not seg:
                continue
            if seg in (["and"], ["a"]):
                out.append(seg[0])
                continue
            n = _words_to_int(seg)
            out.append(str(n) if n is not None else " ".join(seg))
        return " ".join(out)

    t = _PHRASE_RE.sub(phrase, t)
    t = _POINT_RE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", t)
    return t


def parse_money(num: str, unit: str | None) -> int | None:
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return None
    u = (unit or "").lower().strip()
    if u in ("m", "mm", "mil", "million"):
        v *= 1_000_000
    elif u in ("k", "thousand"):
        v *= 1_000
    elif u in ("b", "billion"):
        v *= 1_000_000_000
    return int(round(v))


# ------------------------------------------------------------------ candidates
@dataclass
class Citation:
    message_id: str
    channel: str
    received_at: str
    start: int
    end: int
    quote: str


@dataclass
class Candidate:
    key: str
    value: Any
    confidence: float
    citation: Citation
    correction: bool = False
    order: int = 0


HEDGE_RE = re.compile(r"\b(about|around|roughly|approximately|approx|i think|maybe|probably|give or take|ish|or so|ballpark)\b", re.I)
CORRECTION_RE = re.compile(r"\b(actually|correction|update[d]?|just hired|we now have|now we have|changed|instead|sorry,? (?:i meant|it'?s)|to correct)\b", re.I)
CHANNEL_RELIABILITY = {"email": 1.0, "sms": 0.97, "voice": 1.0}  # voice uses per-segment ASR confidence instead
MONEY = r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|mm|million|thousand|b|billion)?\b"
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE = r"(?:\+?1[\s.-]?)?\(?(\d{3})\)?[\s.-]?(\d{3})[\s.-]?(\d{4})\b"
STREET = (r"(\d{2,6})\s+((?:[A-Z][a-z]+\.?\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|"
          r"Court|Ct|Place|Pl|Parkway|Pkwy))\.?(?:,?\s*(?:Suite|Ste|Unit|#)\s*([\w-]+))?")
ENTITY_SUFFIX = r"(?:LLC|L\.L\.C\.|Inc\.?|Incorporated|Corp\.?|Corporation|Co\.|Company|Ltd\.?|LLP|PLLC)"
NAME_TOKEN = r"[A-Z][\w&'’-]*"


def _norm_phone(a: str, b: str, c: str) -> str:
    return f"+1{a}{b}{c}"


def _parse_date(text: str, ref: date) -> date | None:
    t = text.lower()
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", t)
    if m:
        y = int(m.group(3)) if m.group(3) else ref.year
        y = y + 2000 if y < 100 else y
        d = date(y, int(m.group(1)), int(m.group(2)))
        return d if m.group(3) or d >= ref else date(y + 1, d.month, d.day)
    m = re.search(r"\b(first|1st|15th|fifteenth)\s+of\s+(next month|" + "|".join(MONTHS) + r")\b", t)
    if m:
        day = 1 if m.group(1) in ("first", "1st") else 15
        if m.group(2) == "next month":
            y, mo = (ref.year + (ref.month == 12), ref.month % 12 + 1)
        else:
            mo = MONTHS[m.group(2)]
            y = ref.year if mo > ref.month or (mo == ref.month and day >= ref.day) else ref.year + 1
        return date(y, mo, day)
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b", t)
    if m:
        mo, d = MONTHS[m.group(1)], int(m.group(2))
        y = int(m.group(3)) if m.group(3) else ref.year
        out = date(y, mo, d)
        return out if m.group(3) or out >= ref else date(y + 1, mo, d)
    return None


Rule = Callable[[str, str, str, date], list[tuple[str, Any, float]]]


def _rules() -> list[Rule]:
    rules: list[Rule] = []

    def rule(fn: Rule) -> Rule:
        rules.append(fn)
        return fn

    @rule
    def emails(s, ctx, low, ref):
        return [("contact_email", m.group().lower(), 0.95) for m in re.finditer(EMAIL, s)]

    @rule
    def phones(s, ctx, low, ref):
        if not re.search(r"phone|call|cell|reach|number|tel|mobile|\bp:", low + " " + ctx.lower()) and not re.match(r"^\(?\d", s.strip()):
            return []
        return [("business_phone", _norm_phone(*m.groups()), 0.88) for m in re.finditer(PHONE, s)]

    @rule
    def contact_name(s, ctx, low, ref):
        m = re.search(r"\b(?:my name is|this is|i'?m|i am|speaking[,:]?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z'’-]+)+)", s)
        if m and not re.search(ENTITY_SUFFIX, m.group(1)):
            name = " ".join(m.group(1).split()[:2])
            return [("contact_name", name, 0.85)]
        return []

    @rule
    def applicant(s, ctx, low, ref):
        out = []
        for m in re.finditer(rf"\b((?:{NAME_TOKEN}\s+){{0,5}}{NAME_TOKEN}),?\s+({ENTITY_SUFFIX})(?=[\s,.;!?)]|$)", s):
            words = m.group(1).split()
            while words and words[0].lower() in ("i", "we", "my", "our", "own", "run", "from", "at", "with", "the", "for",
                                                   "called", "named", "is", "this", "hi", "hello", "owner", "of", "and"):
                words = words[1:]
            if words:
                name = " ".join(words) + " " + m.group(2).rstrip(".").replace("L.L.C", "LLC")
                out.append(("applicant_name", name.strip(), 0.92))
                suffix = m.group(2).lower()
                ent = "llc" if "l" == suffix[0] else "corporation" if suffix.startswith(("inc", "corp")) else \
                      "partnership" if suffix.startswith("llp") else "llc" if suffix.startswith("pllc") else "corporation"
                out.append(("entity_type", ent, 0.85))
        return out

    @rule
    def entity(s, ctx, low, ref):
        if re.search(r"sole proprietor", low):
            return [("entity_type", "sole_proprietorship", 0.92)]
        if re.search(r"\bpartnership\b", low):
            return [("entity_type", "partnership", 0.9)]
        if re.search(r"\b(?:s[- ]?corp|c[- ]?corp|corporation)\b", low):
            return [("entity_type", "corporation", 0.9)]
        if re.search(r"\b(?:an? )?llc\b|limited liability company", low) and "entity" in (low + ctx.lower()):
            return [("entity_type", "llc", 0.9)]
        return []

    @rule
    def fein(s, ctx, low, ref):
        m = re.search(r"\b(\d{2})-?(\d{7})\b", s)
        if not m:
            return []
        strong = re.search(r"\b(?:fein|ein|tax id|federal id|employer id)", low + " " + ctx.lower())
        return [("fein", f"{m.group(1)}-{m.group(2)}", 0.95 if strong else 0.6)]

    @rule
    def address(s, ctx, low, ref):
        m = re.search(STREET + r",?\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*),?\s+([A-Z]{2}|" +
                      "|".join(n.title() for n in US_STATES) + r")\.?,?\s+(\d{5})\b", s)
        if not m:
            return []
        street = f"{m.group(1)} {m.group(2).strip()}" + (f" Suite {m.group(3)}" if m.group(3) else "")
        st = m.group(5)
        st = US_STATES.get(st.lower(), st.upper())
        return [("mailing_street", street, 0.93), ("mailing_city", m.group(4), 0.93), ("mailing_state", st, 0.95),
                ("mailing_zip", m.group(6), 0.95)]

    @rule
    def years(s, ctx, low, ref):
        out = []
        m = re.search(r"(?:in business|been (?:open|operating|around|running)|operating|open)\D{0,25}?(\d{1,2})\s+years", low) or \
            (re.search(r"how long|years in business", ctx.lower()) and re.search(r"\b(\d{1,2})\s+years\b", low))
        if m:
            out.append(("years_in_business", int(m.group(1)), 0.9))
        m = re.search(r"\b(?:since|opened in|founded in|started in)\s+((?:19|20)\d{2})\b", low)
        if m:
            out.append(("years_in_business", max(0, ref.year - int(m.group(1))), 0.85))
        return out

    @rule
    def business(s, ctx, low, ref):
        if not re.search(r"\b(we|our|i|my|a|an)\b", low):
            return []
        for key, (kws, desc, naics) in BUSINESS_TYPES.items():
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws):
                return [("business_description", desc, 0.9), ("naics", naics, 0.85)]
        return []

    @rule
    def employees(s, ctx, low, ref):
        out = []
        ft = re.search(r"\b(\d{1,4})\s+(?:are\s+)?full[- ]?time", low) or re.search(r"full[- ]?time\D{0,12}(\d{1,4})\b", low)
        pt = re.search(r"\b(\d{1,4})\s+(?:are\s+)?part[- ]?time", low) or re.search(r"part[- ]?time\D{0,12}(\d{1,4})\b", low)
        if ft:
            out.append(("employees_full_time", int(ft.group(1)), 0.92))
        if pt:
            out.append(("employees_part_time", int(pt.group(1)), 0.92))
        if re.search(r"\bno part[- ]?time|all full[- ]?time|nobody part[- ]?time", low):
            out.append(("employees_part_time", 0, 0.88))
        if not ft:
            m = re.search(r"\b(\d{1,4})[- ](?:person|people|employees?|staff(?: members)?|workers|team members)\b", low) or \
                re.search(r"\bstaff of (\d{1,4})\b|\bteam of (\d{1,4})\b", low)
            if m:
                n = int(next(g for g in m.groups() if g))
                out.append(("_employees_total", n, 0.7))
        return out

    @rule
    def revenue(s, ctx, low, ref):
        both = low + " || " + ctx.lower()
        if not re.search(r"revenue|sales|gross|turnover|bring in|take in|annual(?:ly)?|a year|per year|yearly", both):
            return []
        if re.search(r"occurrence|aggregate|deductible|limit", low):
            return []
        for m in re.finditer(MONEY, s, re.I):
            if not m.group(2) and "$" not in s[max(0, m.start() - 1):m.end()]:
                continue
            v = parse_money(m.group(1), m.group(2))
            if v and v >= 10_000:
                return [("annual_revenue", v, 0.9)]
        return []

    @rule
    def effective(s, ctx, low, ref):
        if not re.search(r"start|effective|begin|coverage|policy|need it by|in place by|when", low + " " + ctx.lower()):
            return []
        d = _parse_date(s, ref)
        return [("policy_effective_date", d.isoformat(), 0.9)] if d else []

    @rule
    def lines(s, ctx, low, ref):
        padded = f" {low} "
        return [("lines_of_business", k, 0.93) for k, kws in LINES.items() if any(kw in padded for kw in kws)]

    @rule
    def limits(s, ctx, low, ref):
        out = []
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(m|million)?\s*/\s*(\d+(?:\.\d+)?)\s*(m|million)\b", low)
        if m and ("limit" in low + ctx.lower() or "occurrence" in low or "aggregate" in low or "/" in low):
            out += [("gl_each_occurrence", parse_money(m.group(1), m.group(2) or m.group(4)), 0.9),
                    ("gl_general_aggregate", parse_money(m.group(3), m.group(4)), 0.9)]
        for m in re.finditer(MONEY + r"\s*(?:per|each|an?)\s*occurrence", low):
            out.append(("gl_each_occurrence", parse_money(m.group(1), m.group(2)), 0.93))
        for m in re.finditer(MONEY + r"\s*(?:general\s*|in\s*(?:the\s*)?)?aggregate", low):
            out.append(("gl_general_aggregate", parse_money(m.group(1), m.group(2)), 0.93))
        return out

    @rule
    def deductible(s, ctx, low, ref):
        if re.search(r"\bno deductible\b|zero deductible", low):
            return [("gl_deductible", 0, 0.9)]
        m = re.search(MONEY + r"\s*(?:dollars?\s+)?deductible", low) or re.search(r"deductible\D{0,15}" + MONEY, low)
        if m:
            return [("gl_deductible", parse_money(m.group(1), m.group(2)), 0.9)]
        return []

    @rule
    def losses(s, ctx, low, ref):
        both = low + " " + ctx.lower()
        if re.search(r"\bno (?:prior )?(?:claims|losses)\b|never (?:had|filed) a claim|zero claims|claims? free|no, none", low):
            return [("prior_losses_3y", 0, 0.9)]
        m = re.search(r"\b(\d{1,2})\s+(?:small\s+|minor\s+)?(?:claims?|losses)\b", low)
        if m:
            return [("prior_losses_3y", int(m.group(1)), 0.88)]
        if "claim" in ctx.lower() and re.search(r"^\s*(?:no|none|nope)\b", low):
            return [("prior_losses_3y", 0, 0.85)]
        return []

    @rule
    def carrier(s, ctx, low, ref):
        if re.search(r"no (?:current )?(?:insurance|carrier|coverage)|first time (?:getting|buying)|never been insured|not insured", low):
            return [("prior_carrier", "none", 0.85)]
        m = re.search(r"(?:currently (?:with|insured (?:with|by))|carrier is|insured with|our (?:current )?(?:insurer|carrier) is|"
                      r"policy (?:is )?with)\s+((?:[A-Z][\w&]*\s?){1,4})", s)
        if m:
            return [("prior_carrier", m.group(1).strip().rstrip("."), 0.85)]
        return []

    @rule
    def hazards(s, ctx, low, ref):
        out = []
        for key, (kws, _) in HAZARDS.items():
            for kw in kws:
                i = low.find(kw)
                if i >= 0 and not re.search(r"\b(?:no|not|don'?t|never|without)\b[\w\s]{0,15}$", low[:i]):
                    out.append(("hazards", key, 0.9))
                    break
        return out

    return rules


RULES = _rules()


# ------------------------------------------------------------------ resolution
@dataclass
class FieldResult:
    key: str
    value: Any
    confidence: float
    status: str            # confident | low | conflict | missing
    citations: list[Citation] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)


@dataclass
class CaseExtraction:
    fields: dict[str, FieldResult]

    def value(self, key: str) -> Any:
        f = self.fields.get(key)
        return f.value if f else None

    @property
    def high_risk_hazards(self) -> list[str]:
        return sorted(h for h in (self.value("hazards") or []) if h in HIGH_RISK)

    def gaps(self) -> list[str]:
        return [k for k in REQUIRED if self.fields[k].status in ("missing", "low")]

    def conflicts(self) -> list[str]:
        return [k for k in REQUIRED if self.fields[k].status == "conflict"]

    @property
    def ready(self) -> bool:
        return not self.gaps() and not self.conflicts()

    def record(self) -> dict:
        return {k: f.value for k, f in self.fields.items() if f.value is not None}

    def to_dict(self) -> dict:
        return {k: {"value": f.value, "confidence": round(f.confidence, 3), "status": f.status,
                    "citations": [asdict(c) for c in f.citations], "alternatives": f.alternatives}
                for k, f in self.fields.items()}


def _noisy_or(cs: Iterable[float]) -> float:
    p = 1.0
    for c in cs:
        p *= 1 - c
    return min(0.99, 1 - p)


def candidates(messages: list[InboundMessage]) -> list[Candidate]:
    out: list[Candidate] = []
    order = 0
    for msg in messages:
        ref = datetime.fromisoformat(msg.received_at).date()
        rel = CHANNEL_RELIABILITY.get(msg.channel, 0.9)
        for sent in msg.sentences:
            norm = normalize_numbers(sent.text)
            low = norm.lower()
            hedge = 0.92 if HEDGE_RE.search(low) else 1.0
            corr = bool(CORRECTION_RE.search(low))
            cite = Citation(msg.message_id, msg.channel, msg.received_at, sent.start, sent.end, sent.text)
            for rule in RULES:
                for key, value, base in rule(norm, normalize_numbers(sent.context), low, ref):
                    if value is None:
                        continue
                    order += 1
                    out.append(Candidate(key, value, base * rel * sent.asr_confidence * hedge, cite, corr, order))
        if msg.channel == "email" and "@" in msg.contact:
            order += 1
            out.append(Candidate("contact_email", msg.contact, 0.93,
                                 Citation(msg.message_id, "email", msg.received_at, 0, 0, f"From: <{msg.contact}>"), False, order))
        if msg.channel == "email" and msg.meta.get("from_name"):
            order += 1
            out.append(Candidate("contact_name", msg.meta["from_name"], 0.93,
                                 Citation(msg.message_id, "email", msg.received_at, 0, 0, f"From: {msg.meta['from_name']}"), False, order))
        if msg.channel in ("sms", "voice") and msg.contact.startswith("+"):
            order += 1
            out.append(Candidate("business_phone", msg.contact, 0.8,
                                 Citation(msg.message_id, msg.channel, msg.received_at, 0, 0, f"caller ID {msg.contact}"), False, order))
    return out


def resolve(cands: list[Candidate]) -> CaseExtraction:
    by_key: dict[str, list[Candidate]] = {}
    for c in cands:
        by_key.setdefault(c.key, []).append(c)
    # employees: a bare headcount only fills full-time when no explicit split exists
    if "_employees_total" in by_key and "employees_full_time" not in by_key:
        by_key["employees_full_time"] = [Candidate("employees_full_time", c.value, c.confidence, c.citation, c.correction, c.order)
                                         for c in by_key["_employees_total"]]
    fields: dict[str, FieldResult] = {}
    for spec in FIELDS:
        cs = sorted(by_key.get(spec.key, []), key=lambda c: c.order)
        if not cs:
            fields[spec.key] = FieldResult(spec.key, None, 0.0, "missing")
            continue
        if spec.kind == "list":
            vals: dict[str, list[Candidate]] = {}
            for c in cs:
                vals.setdefault(c.value, []).append(c)
            conf = min(_noisy_or(c.confidence for c in v) for v in vals.values())
            fields[spec.key] = FieldResult(spec.key, sorted(vals), conf, "confident" if conf >= CONFIDENCE_GATE else "low",
                                           [v[0].citation for v in vals.values()])
            continue
        groups: dict[str, list[Candidate]] = {}
        for c in cs:
            groups.setdefault(str(c.value).lower(), []).append(c)
        corrections = [c for c in cs if c.correction]
        if len(groups) > 1 and corrections and str(corrections[-1].value).lower() != str(cs[0].value).lower():
            win = groups[str(corrections[-1].value).lower()]
            conf = _noisy_or(c.confidence for c in win)
            alts = [{"value": g[0].value, "superseded_by_correction": True, "quote": g[0].citation.quote}
                    for k, g in groups.items() if k != str(win[0].value).lower()]
            fields[spec.key] = FieldResult(spec.key, win[-1].value, conf, "confident" if conf >= CONFIDENCE_GATE else "low",
                                           [c.citation for c in win], alts)
            continue
        ranked = sorted(groups.values(), key=lambda g: (_noisy_or(c.confidence for c in g), g[-1].order), reverse=True)
        win = ranked[0]
        conf = _noisy_or(c.confidence for c in win)
        alts = [{"value": g[0].value, "confidence": round(_noisy_or(c.confidence for c in g), 3), "quote": g[0].citation.quote}
                for g in ranked[1:]]
        status = "confident" if conf >= CONFIDENCE_GATE else "low"
        if alts and spec.key not in ("business_phone", "contact_name", "naics"):
            conf, status = conf * 0.5, "conflict"
        fields[spec.key] = FieldResult(spec.key, win[-1].value, conf, status, [c.citation for c in win], alts)
    if fields["employees_part_time"].status == "missing" and fields["employees_full_time"].status == "confident" \
            and "_employees_total" not in by_key:
        f = fields["employees_full_time"]
        fields["employees_part_time"] = FieldResult("employees_part_time", 0, 0.6, "low", f.citations[:1])
    return CaseExtraction(fields)


def extract(messages: list[InboundMessage], llm=None) -> CaseExtraction:
    cands = candidates(messages)
    if llm is not None:
        cands += llm_candidates(messages, llm)
    return resolve(cands)


def llm_candidates(messages: list[InboundMessage], llm) -> list[Candidate]:
    """Optional LLM pass. A proposal survives only if its quote is verbatim in a source message."""
    import json as _json

    from core.harness.llm import Message, MockLLM

    if llm is None or isinstance(llm, MockLLM):
        return []
    corpus = "\n\n".join(f"[{m.message_id} {m.channel}] {m.text}" for m in messages)
    prompt = ("Extract ACORD 125/126 fields from these customer messages. Reply ONLY with JSON "
              '{"field_key": {"value": ..., "quote": "<exact text copied from a message>"}} using keys '
              f"{[f.key for f in FIELDS]}.\n\n{corpus[:12000]}")
    try:
        resp = llm.complete([Message("user", prompt)], [], 1200)
        data = _json.loads(re.search(r"\{.*\}", resp.text, re.S).group())
    except Exception:
        return []
    out = []
    for key, item in data.items():
        if key not in SPEC or not isinstance(item, dict) or not item.get("quote"):
            continue
        for m in messages:
            i = m.text.find(item["quote"])
            if i >= 0:
                out.append(Candidate(key, item.get("value"), 0.8, Citation(m.message_id, m.channel, m.received_at, i,
                                                                            i + len(item["quote"]), item["quote"]), False, 10**6))
                break
    return out
