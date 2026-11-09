"""
================================================================================
BLOCK: Synthetic Intake Benchmark (50 submissions + customer simulator)
--------------------------------------------------------------------------------
Role & contracts
  Generates deterministic commercial-insurance intake cases with GROUND
  TRUTH ACORD-125/126 records and raw channel payloads exactly as the
  providers deliver them:
    day 1  e-mail      RFC 5322 text (headers, body, quoted trail, signature)
    day 2  Twilio SMS  webhook form fields + a valid X-Twilio-Signature
    day 3  voice call  diarised transcript JSON (agent/caller, ASR confidence,
                       numbers spoken as words)
  plus a CustomerSimulator that answers the agent's follow-up questions on
  the channel the customer uses that day.
  Consumers: scripts/run_soma_eval.py and tests.

Key engineering mechanism
  Scenario mix (seeded): clean 18, needs_followup 14 (required facts held
  back until asked), conflict_corrected 6 (headcount changes mid-thread with
  a correction cue), high_risk 7 (liquor, roof work, childcare, explosives ->
  must hand off), ambiguous 5 (revenue never pinned down -> must hand off).
  Phrasing is drawn from several templates per field and channel so the
  extractor is not tested on a single sentence form.

Failure modes & 1-step mitigation
  * Benchmark/extractor co-evolution (overfitting) is a real risk with
    synthetic data. Templates vary per field, voice uses spoken numbers and
    ASR noise, and replacing `build_cases()` with real transcripts needs no
    other code change.
  * All names, businesses, e-mail domains (*.example) and phone numbers
    (555-01xx) are fictional.
================================================================================
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import format_datetime

from agents.soma_intake.extractor import BUSINESS_TYPES

TWILIO_AUTH_TOKEN = "test-auth-token-not-a-secret"
SMS_WEBHOOK_URL = "https://soma.example/soma/inbound/sms"
AGENCY_PHONE = "+18005550100"

FIRST = ["Rosa", "Daniel", "Priya", "Marcus", "Elena", "Tomas", "Grace", "Hiro", "Amara", "Luca", "Nadia", "Owen",
         "Sofia", "Kenji", "Leah", "Mateo", "Ivy", "Samuel", "Zoe", "Rafael"]
LAST = ["Martinez", "Okafor", "Shah", "Lindqvist", "Moreau", "Novak", "Kim", "Tanaka", "Diallo", "Rossi", "Haddad",
        "Brennan", "Alvarez", "Sato", "Cohen", "Ferreira", "Walsh", "Mensah", "Park", "Duarte"]
NAME_PARTS = ["Sweet Crumb", "Golden Oak", "Blue Fern", "Harbor Light", "Copper Kettle", "Maple Row", "North Star",
              "Silver Birch", "Red Lantern", "Green Leaf", "Stonebridge", "Bright Path", "Juniper", "Cedar Point", "Lark Hill"]
TYPE_NOUN = {"bakery": "Bakery", "coffee_shop": "Coffee", "florist": "Flowers", "landscaping": "Landscaping",
             "it_consulting": "IT Partners", "dental_office": "Dental", "yoga_studio": "Yoga", "cleaning_service": "Cleaning",
             "auto_repair": "Auto Repair", "bookstore": "Books", "food_truck": "Street Eats", "photography": "Photography",
             "bar": "Tavern", "roofing": "Roofing", "daycare": "Little Learners", "fireworks": "Fireworks", "brewery": "Brewing"}
SAFE_TYPES = ["bakery", "coffee_shop", "florist", "landscaping", "it_consulting", "dental_office", "yoga_studio",
              "cleaning_service", "auto_repair", "bookstore", "food_truck", "photography"]
HIGH_RISK_TYPES = {"bar": ("liquor", "We serve beer and wine and run a full bar on weekends."),
                   "roofing": ("roof_work", "Our crews work on rooftops most days."),
                   "daycare": ("childcare", "We look after toddlers and infants during the day."),
                   "fireworks": ("explosives", "We sell consumer fireworks out of the shop."),
                   "brewery": ("liquor", "We pour beer in the taproom every evening.")}
MILD_HAZARD_TEXT = {"deep_fryer": "We fry donuts in a deep fryer every morning.",
                    "delivery": "We do deliveries with one van.",
                    "offsite_catering": "We also cater events a few times a month.",
                    "power_tools": "The crew uses mowers and other power tools."}
CITIES = [("San Francisco", "CA", "941"), ("Oakland", "CA", "946"), ("Austin", "TX", "787"), ("Denver", "CO", "802"),
          ("Portland", "OR", "972"), ("Seattle", "WA", "981"), ("Chicago", "IL", "606"), ("Boston", "MA", "021")]
STATE_NAMES = {"CA": "California", "TX": "Texas", "CO": "Colorado", "OR": "Oregon", "WA": "Washington", "IL": "Illinois",
               "MA": "Massachusetts"}
STREETS = ["Larkin Street", "Valencia Street", "Market Street", "Oak Avenue", "Pine Street", "Cedar Lane", "Elm Drive",
           "Harbor Way", "Mission Street", "Grand Avenue"]
CARRIERS = ["Harborview Mutual", "Pinecrest Insurance", "Northgate Casualty", "Bluewater Assurance"]
SCENARIOS = ["clean"] * 18 + ["needs_followup"] * 14 + ["conflict_corrected"] * 6 + ["high_risk"] * 7 + ["ambiguous"] * 5

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))


def money_words(v: int) -> str:
    if v >= 1_000_000:
        m = v / 1_000_000
        return f"{num_words(int(m))} point {num_words(int(round((m - int(m)) * 10)))} million" if m != int(m) else f"{num_words(int(m))} million"
    return f"{num_words(v // 1000)} thousand"


def money_short(v: int) -> str:
    return f"${v / 1_000_000:g}M" if v >= 1_000_000 else f"${v // 1000}k"


@dataclass
class Case:
    case_no: int
    scenario: str
    truth: dict
    expect_handoff: bool
    payloads: list[dict] = field(default_factory=list)  # [{"day", "channel", "payload"}]
    vague_fields: list[str] = field(default_factory=list)
    style: str = "dev"   # dev = phrasing used while building the rules; heldout = never seen by them


def twilio_signature(url: str, params: dict, token: str = TWILIO_AUTH_TOKEN) -> str:
    data = url + "".join(k + params[k] for k in sorted(params))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def sms_payload(rng: random.Random, frm: str, body: str) -> dict:
    form = {"MessageSid": "SM" + "%032x" % rng.getrandbits(128), "AccountSid": "AC" + "0" * 32, "From": frm,
            "To": AGENCY_PHONE, "Body": body, "NumMedia": "0"}
    return {"form": form, "signature": twilio_signature(SMS_WEBHOOK_URL, form), "url": SMS_WEBHOOK_URL}


def email_payload(rng: random.Random, truth: dict, when: datetime, subject: str, body: str, quoted: str = "") -> dict:
    domain = truth["contact_email"].split("@")[1]
    raw = (f"From: {truth['contact_name']} <{truth['contact_email']}>\nTo: quotes@soma.example\nSubject: {subject}\n"
           f"Date: {format_datetime(when)}\nMessage-ID: <{rng.getrandbits(48):x}@{domain}>\nMIME-Version: 1.0\n"
           f"Content-Type: text/plain; charset=utf-8\n\n{body}\n")
    if quoted:
        raw += "\nOn Mon, Soma Intake <quotes@soma.example> wrote:\n" + "\n".join("> " + l for l in quoted.splitlines()) + "\n"
    return {"raw": raw}


def voice_payload(rng: random.Random, frm: str, when: datetime, turns: list[tuple[str, str]], low_asr: bool = False) -> dict:
    segs, t = [], 0.0
    for speaker, text in turns:
        dur = 1.5 + len(text) / 14
        conf = round(rng.uniform(0.84, 0.90) if low_asr else rng.uniform(0.90, 0.98), 3)
        segs.append({"speaker": speaker, "start": round(t, 2), "end": round(t + dur, 2), "text": text,
                     "confidence": 1.0 if speaker == "agent" else conf})
        t += dur + 0.4
    return {"call_sid": "CA" + "%032x" % rng.getrandbits(128), "from": frm, "to": AGENCY_PHONE,
            "started_at": when.isoformat(), "segments": segs}


# ------------------------------------------------------------------ HELD-OUT phrasing (never used to develop the rules)
HELDOUT_HIGH_RISK = {"bar": "We have a liquor license and mix cocktails at night.",
                     "roofing": "Mostly we replace shingles on houses.",
                     "daycare": "Parents drop their kids off with us all day.",
                     "fireworks": "We sell pyrotechnics for the holidays.",
                     "brewery": "Guests drink our beer on site."}


def say_heldout(key: str, t: dict, channel: str, rng: random.Random) -> str:
    v = t.get(key)
    spoken = channel == "voice"
    amt = lambda x: money_words(x) if spoken else money_short(x)
    if key == "fein":
        return rng.choice([f"Federal tax ID number: {v}.", f"Our employer identification number is {v}."])
    if key == "address":
        st = STATE_NAMES[t["mailing_state"]] if spoken else t["mailing_state"]
        zip_ = " ".join(_ONES[int(d)] for d in t["mailing_zip"]) if spoken else t["mailing_zip"]
        return rng.choice([f"Our shop is located at {t['mailing_street']}, {t['mailing_city']}, {st} {zip_}.",
                           f"Send mail to {t['mailing_street']}, {t['mailing_city']}, {st} {zip_}."])
    if key == "annual_revenue":
        return rng.choice([f"Yearly turnover is roughly {amt(v)}.", f"We brought in {amt(v)} last year.",
                           f"Top line is about {amt(v)} annually."])
    if key == "policy_effective_date":
        d = date.fromisoformat(v)
        words = f"{d.strftime('%B')} {d.day}" if spoken else f"{d.month}/{d.day}"
        return rng.choice([f"Coverage needs to begin on {words}.", f"Can the policy kick in {words}?"])
    if key == "limits":
        o, a = t["gl_each_occurrence"], t["gl_general_aggregate"]
        return rng.choice([f"Looking for {amt(o)} per claim and {amt(a)} total.",
                           f"{amt(o)} each occurrence, {amt(a)} general aggregate."])
    if key == "gl_deductible":
        return "We'd rather not have a deductible." if v == 0 else f"We'd take a {num_words(v) + ' dollar' if spoken else '$' + format(v, ',')} deductible."
    if key == "prior_losses_3y":
        return "Zero losses so far." if v == 0 else f"We've filed {num_words(v) if spoken else v} claims since 2023."
    if key == "prior_carrier":
        return "Never had insurance before." if v == "none" else f"Our policy is with {v} right now."
    if key == "employees_split":
        ft, pt = t["employees_full_time"], t["employees_part_time"]
        f_, p_ = (num_words(ft), num_words(pt)) if spoken else (ft, pt)
        return f"We employ {f_} people full time plus {p_} part timers." if pt else f"Everyone is full time, {f_} people."
    if key == "years_in_business":
        return rng.choice([f"Been around since {2026 - v}.", f"We're {num_words(v) if spoken else v} years old as a business."])
    if key == "entity_type":
        return {"llc": "Legally we're a limited liability company.", "corporation": "Legally we're set up as a corporation.",
                "sole_proprietorship": "It's just me as a sole proprietor.", "partnership": "Two of us own it as partners."}[v]
    if key == "lines_of_business":
        return "We want a GL policy" + (" plus property insurance." if "property" in v else ".")
    raise KeyError(key)


# ------------------------------------------------------------------ phrasing per field (statement on a channel)
def say(key: str, t: dict, channel: str, rng: random.Random, style: str = "dev") -> str:
    if style == "heldout":
        return say_heldout(key, t, channel, rng)
    v = t.get(key)
    spoken = channel == "voice"
    if key == "fein":
        return rng.choice([f"Our EIN is {v}.", f"FEIN {v}.", f"Tax ID is {v}."])
    if key == "address":
        st = STATE_NAMES[t["mailing_state"]] if spoken else t["mailing_state"]
        zip_ = " ".join(_ONES[int(d)] for d in t["mailing_zip"]) if spoken else t["mailing_zip"]
        return rng.choice([f"We're at {t['mailing_street']}, {t['mailing_city']}, {st} {zip_}.",
                           f"Mailing address is {t['mailing_street']}, {t['mailing_city']}, {st} {zip_}."])
    if key == "annual_revenue":
        amt = money_words(v) if spoken else money_short(v)
        return rng.choice([f"We do about {amt} in sales a year.", f"Annual revenue is around {amt}.",
                           f"Gross sales last year were {amt}."])
    if key == "policy_effective_date":
        d = date.fromisoformat(v)
        words = f"{d.strftime('%B')} {d.day}" if spoken else rng.choice([f"{d.month}/{d.day}/{d.year}", d.strftime("%B %-d, %Y")])
        return rng.choice([f"We'd like coverage to start {words}.", f"Effective date {words} please.",
                           f"We need the policy in place by {words}."])
    if key == "limits":
        o, a = t["gl_each_occurrence"], t["gl_general_aggregate"]
        if spoken:
            return f"{money_words(o)} per occurrence and {money_words(a)} aggregate."
        return rng.choice([f"Limits {money_short(o)} per occurrence, {money_short(a)} aggregate.",
                           f"{o // 1_000_000}M/{a // 1_000_000}M limits."])
    if key == "gl_deductible":
        return "No deductible please." if v == 0 else (f"A {num_words(v)} dollar deductible is fine." if spoken
                                                       else f"${v:,} deductible is fine.")
    if key == "prior_losses_3y":
        return "No claims in the last three years." if v == 0 else (f"We had {num_words(v)} small claims in the past three years."
                                                                    if spoken else f"We had {v} small claims in the past 3 years.")
    if key == "prior_carrier":
        return "This is our first time buying insurance." if v == "none" else f"We're currently insured with {v}."
    if key == "employees_split":
        ft, pt = t["employees_full_time"], t["employees_part_time"]
        f_, p_ = (num_words(ft), num_words(pt)) if spoken else (ft, pt)
        return f"{f_} full-time and {p_} part-time." if pt else f"{f_} full-time, no part-time staff."
    if key == "years_in_business":
        return rng.choice([f"We've been in business {num_words(v) if spoken else v} years.",
                           f"We opened in {2026 - v}."])
    if key == "entity_type":
        return {"llc": "We're an LLC.", "corporation": "We're incorporated as an S corp.",
                "sole_proprietorship": "I'm a sole proprietor.", "partnership": "We're a partnership."}[v]
    if key == "lines_of_business":
        return "We need general liability" + (" and property coverage." if "property" in v else " coverage.")
    raise KeyError(key)


# fields the follow-up question keys map to (what the customer will state)
ASK_TO_STATEMENT = {"mailing_street": "address", "mailing_city": "address", "mailing_state": "address",
                    "mailing_zip": "address", "gl_each_occurrence": "limits", "gl_general_aggregate": "limits",
                    "employees_full_time": "employees_split", "employees_part_time": "employees_split"}


def build_case(n: int, scenario: str, rng: random.Random, start: date, style: str = "dev") -> Case:
    btype = rng.choice(list(HIGH_RISK_TYPES)) if scenario == "high_risk" else rng.choice(SAFE_TYPES)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    part = rng.choice(NAME_PARTS)
    entity = rng.choice(["llc", "llc", "corporation", "sole_proprietorship"]) if scenario != "high_risk" else "llc"
    suffix = {"llc": "LLC", "corporation": "Inc", "sole_proprietorship": "LLC", "partnership": "LLP"}[entity]
    if entity == "sole_proprietorship":
        entity = "llc"  # a sole prop would not carry a suffix; keep the record self-consistent
    city, state, zip3 = rng.choice(CITIES)
    ft = rng.randint(3, 25)
    pt = rng.choice([0, 0, 1, 2, 3, 4])
    domain = (part + TYPE_NOUN[btype]).lower().replace(" ", "") + ".example"
    eff = start + timedelta(days=rng.choice([20, 30, 35, 45, 60]))
    occ = rng.choice([1_000_000, 1_000_000, 2_000_000])
    hazards = []
    if scenario == "high_risk":
        hazards.append(HIGH_RISK_TYPES[btype][0])
    for hz in ("deep_fryer", "delivery", "offsite_catering", "power_tools"):
        if (btype in ("bakery", "food_truck") and hz == "deep_fryer" and rng.random() < 0.5) or \
           (hz == "delivery" and btype in ("bakery", "florist") and rng.random() < 0.5) or \
           (hz == "power_tools" and btype == "landscaping"):
            hazards.append(hz)
    truth = {
        "applicant_name": f"{part} {TYPE_NOUN[btype]} {suffix}", "entity_type": entity,
        "fein": f"{rng.randint(10, 99)}-{rng.randint(1_000_000, 9_999_999)}",
        "mailing_street": f"{rng.randint(100, 4999)} {rng.choice(STREETS)}", "mailing_city": city, "mailing_state": state,
        "mailing_zip": f"{zip3}{rng.randint(10, 99)}", "business_phone": f"+1415555{100 + n:04d}",  # unique per case (555-01xx fictional range)
        "contact_name": f"{first} {last}", "contact_email": f"{first.lower()}@{domain}",
        "years_in_business": rng.randint(2, 18), "business_description": BUSINESS_TYPES[btype][1],
        "naics": BUSINESS_TYPES[btype][2], "employees_full_time": ft, "employees_part_time": pt,
        "annual_revenue": rng.choice([350_000, 600_000, 850_000, 1_200_000, 1_500_000, 2_400_000, 3_100_000]),
        "policy_effective_date": eff.isoformat(),
        "lines_of_business": sorted(["general_liability"] + (["property"] if rng.random() < 0.3 else [])),
        "gl_each_occurrence": occ, "gl_general_aggregate": occ * 2, "gl_deductible": rng.choice([0, 500, 1000, 2500]),
        "prior_losses_3y": rng.choice([0, 0, 0, 1, 2]), "prior_carrier": rng.choice(CARRIERS + ["none"]),
        "hazards": sorted(hazards), "_business_type": btype,
    }
    case = Case(n, scenario, truth, expect_handoff=scenario in ("high_risk", "ambiguous"), style=style)
    S = lambda key, ch: say(key, truth, ch, rng, style)
    noun = BUSINESS_TYPES[btype][0][0]
    headcount = ft + pt
    if scenario == "conflict_corrected":
        truth["employees_full_time"] = ft = ft + 2   # hired two more between e-mail and call
    stmts = {1: ["entity_type", "lines_of_business", "years_in_business"],
             2: ["fein", "annual_revenue", "policy_effective_date"],
             3: ["limits", "gl_deductible", "prior_losses_3y", "prior_carrier", "employees_split"]}
    (stmts[1] if rng.random() < 0.5 else stmts[2]).append("address")
    held: list[str] = []
    if scenario == "needs_followup":
        pool = ["address", "policy_effective_date", "limits", "annual_revenue", "employees_split"]
        held = rng.sample(pool, rng.randint(1, 2))
        for d in stmts:
            stmts[d] = [s for s in stmts[d] if s not in held]
    if scenario == "ambiguous":
        stmts[2] = [s for s in stmts[2] if s != "annual_revenue"]
        case.vague_fields = ["annual_revenue"]
    case.truth["_held_back"] = held

    d1 = datetime.combine(start, datetime.min.time(), tzinfo=timezone(timedelta(hours=-8))) + timedelta(hours=9, minutes=rng.randint(0, 50))
    intro = (f"Hello, {first} {last} here. I run {truth['applicant_name']}, a small {noun} in {city} with {headcount} employees."
             if style == "heldout" else None) or rng.choice([f"My name is {first} {last} and I own {truth['applicant_name']}, a {headcount}-person {noun} in {city}.",
                        f"I'm {first} {last}, owner of {truth['applicant_name']}. We're a {noun} in {city} with a team of {headcount}."])
    body = [f"Hi there,", "", intro, S("lines_of_business", "email")]
    body += [S(k, "email") for k in stmts[1] if k != "lines_of_business"]
    if scenario == "high_risk":
        body.append(HELDOUT_HIGH_RISK[btype] if style == "heldout" else HIGH_RISK_TYPES[btype][1])
    for hz in hazards:
        if hz in MILD_HAZARD_TEXT:
            body.append(MILD_HAZARD_TEXT[hz])
    body += ["", "Thanks,", f"{first} {last}", f"Owner, {truth['applicant_name']}",
             f"Phone: ({truth['business_phone'][2:5]}) {truth['business_phone'][5:8]}-{truth['business_phone'][8:]}"]
    case.payloads.append({"day": 1, "channel": "email", "payload": email_payload(rng, truth, d1, f"Insurance quote for our {noun}", "\n".join(body))})

    sms_lines = [S(k, "sms") for k in stmts[2]]
    if scenario == "ambiguous":
        sms_lines.append("Revenue honestly depends on the season, hard to say.")
    sms_body = (f"{first} @ {part} here. " if style == "heldout" else f"Hi it's {first} from {part}. ") + " ".join(sms_lines)
    case.payloads.append({"day": 2, "channel": "sms", "payload": sms_payload(rng, truth["business_phone"], sms_body)})

    turns = [("agent", f"Thanks for calling Soma, this is Jordan. Who am I speaking with?"),
             ("caller", f"Hi, this is {first} {last} from {truth['applicant_name']}.")]
    q = {"limits": "What liability limits are you looking for?", "gl_deductible": "Any preference on the deductible?",
         "prior_losses_3y": "Any claims or losses in the last three years?", "prior_carrier": "Who are you insured with today?",
         "employees_split": "How many of your staff are full-time versus part-time?", "address": "What's the mailing address?",
         "annual_revenue": "Roughly what's your annual revenue?", "policy_effective_date": "When should coverage start?",
         "fein": "Do you have your EIN handy?"}
    for k in stmts[3]:
        turns.append(("agent", q[k]))
        if k == "employees_split" and scenario == "conflict_corrected":
            turns.append(("caller", f"Actually we just hired two more, so we now have {num_words(ft)} full-time and "
                                    f"{num_words(truth['employees_part_time'])} part-time." if truth["employees_part_time"] else
                                    f"Actually we just hired two more, so we now have {num_words(ft)} full-time, no part-time staff."))
        else:
            turns.append(("caller", S(k, "voice")))
    turns.append(("agent", "Great, I'll get this to underwriting."))
    case.payloads.append({"day": 3, "channel": "voice",
                          "payload": voice_payload(rng, truth["business_phone"], d1 + timedelta(days=2, hours=5), turns,
                                                   low_asr=scenario == "ambiguous")})
    if scenario == "conflict_corrected":
        case.truth["_initial_headcount"] = headcount
    return case


def build_cases(n: int = 50, seed: int = 2026, start: date = date(2026, 11, 2), style: str = "dev") -> list[Case]:
    rng = random.Random(seed)
    scen = SCENARIOS[:n] if n <= len(SCENARIOS) else (SCENARIOS * (n // len(SCENARIOS) + 1))[:n]
    rng.shuffle(scen)
    return [build_case(i + 1, s, rng, start + timedelta(days=i % 5), style) for i, s in enumerate(scen)]


class CustomerSimulator:
    """Answers the agent's follow-up questions from the case's ground truth (vague where the scenario says so)."""

    def __init__(self, case: Case, seed: int = 0):
        self.case, self.rng = case, random.Random(seed * 1000 + case.case_no)

    def reply(self, asked_fields: list[str], channel: str, when: datetime) -> dict:
        t = self.case.truth
        statements, seen = [], set()
        for f in asked_fields:
            st = ASK_TO_STATEMENT.get(f, f)
            if st in seen:
                continue
            seen.add(st)
            if f in self.case.vague_fields:
                statements.append(self.rng.choice(["Honestly not sure, it really depends on the season.",
                                                   "Hard to say, we're still figuring that out."]))
                continue
            try:
                statements.append(say(st, t, "voice" if channel == "voice" else "sms", self.rng, self.case.style))
            except KeyError:
                continue
        text = " ".join(statements) or "Sorry, not sure what you need."
        if channel == "sms":
            return sms_payload(self.rng, t["business_phone"], text)
        if channel == "email":
            return email_payload(self.rng, t, when, "Re: your questions", text)
        return voice_payload(self.rng, t["business_phone"], when,
                             [("agent", "Following up on your application, I have a couple of questions."), ("caller", text)])
