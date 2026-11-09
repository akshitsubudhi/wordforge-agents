"""
================================================================================
BLOCK: Mock Carrier Portal + Label-Driven Portal Automation
--------------------------------------------------------------------------------
Role & contracts
  portal_router   FastAPI routes for a mock carrier's quote-intake portal:
                    GET  /mock-carrier-portal         HTML form (ACORD 125 + 126 sections)
                    POST /mock-carrier-portal/submit  server-side validation, then
                                                      a confirmation page with a submission id
                  Mounted on the main app (app/main.py) and also available
                  standalone (portal_app()) for in-process automation.
  PortalClient    fills that form from an extracted ACORD record the way an
                  RPA bot would: fetch the page, read the <label>s, map labels
                  to ACORD fields, format values per input type, post, and
                  parse the confirmation or the per-field errors.
  Downstream: carrier_portal.sqlite (submissions), read by the evaluator
              to check the carrier received the TRUE values.

Key engineering mechanism
  * Anti-brittleness by design: field NAMES are HMAC-derived from a
    per-form nonce (they change every page load), so a bot that hard-codes
    selectors fails. The client maps fields by LABEL TEXT: an explicit
    synonym table, then a token-Jaccard fallback.
  * Signed form token: CSRF protection and the key for name de-obfuscation.
  * Server-side validation: required fields, ZIP / state / e-mail formats,
    money and limit sets, aggregate >= occurrence.
  * Idempotency: the hidden client_ref (the workflow's activity id) is
    UNIQUE, so a retried submission after a crash returns the original
    submission id instead of filing twice.

Failure modes & 1-step mitigation
  * Portal rejects a value -> FillResult carries field-level errors; the
    workflow routes the case to a human with those errors (no silent loss).
  * Portal relabels a field -> the fuzzy label mapping still finds it; a
    required field it cannot map is reported in `unmapped_required`.
  * Network retry / crash mid-submit -> same client_ref, same submission id.
================================================================================
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import re
import secrets
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse

from core.config import get_settings

PORTAL_URL = "https://carrier.mock/mock-carrier-portal"
_SECRET = secrets.token_bytes(32)
STATES = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA",
          "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
          "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
LIMITS = [500_000, 1_000_000, 2_000_000, 3_000_000, 4_000_000]

# portal field definitions: (portal label, internal id, input type, required, options)
PORTAL_FIELDS = [
    ("ACORD 125 - Applicant", None, "section", False, None),
    ("Named Insured (Applicant Name)", "insured", "text", True, None),
    ("Legal Entity", "entity", "select", True, [("llc", "Limited Liability Company"), ("corporation", "Corporation"),
                                                  ("partnership", "Partnership"), ("sole_proprietorship", "Individual / Sole Proprietor")]),
    ("FEIN / Tax ID", "fein", "text", False, None),
    ("Street Address", "street", "text", True, None),
    ("City", "city", "text", True, None),
    ("State", "state", "select", True, [(s, s) for s in STATES]),
    ("ZIP", "zip", "text", True, None),
    ("Business Phone", "phone", "tel", False, None),
    ("Contact Person", "contact", "text", True, None),
    ("Contact E-mail", "email", "email", True, None),
    ("Years in Business", "years", "number", False, None),
    ("Description of Operations", "ops", "text", True, None),
    ("NAICS", "naics", "text", False, None),
    ("# Full-Time Employees", "ft", "number", True, None),
    ("# Part-Time Employees", "pt", "number", False, None),
    ("Annual Gross Sales ($)", "sales", "number", True, None),
    ("Requested Effective Date", "eff", "date", True, None),
    ("Lines of Coverage", "lines", "checkbox", True, [("general_liability", "General Liability"), ("property", "Property"),
                                                      ("workers_comp", "Workers Compensation"), ("commercial_auto", "Commercial Auto")]),
    ("ACORD 126 - Commercial General Liability", None, "section", False, None),
    ("Each Occurrence Limit ($)", "occ", "select", True, [(str(v), f"${v:,}") for v in LIMITS]),
    ("General Aggregate Limit ($)", "agg", "select", True, [(str(v), f"${v:,}") for v in LIMITS]),
    ("Deductible ($)", "ded", "number", False, None),
    ("Losses (past 3 yrs)", "losses", "number", False, None),
    ("Current / Prior Carrier", "carrier", "text", False, None),
]


def _db() -> sqlite3.Connection:
    path = get_settings().data_dir / "carrier_portal.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS submissions (submission_id TEXT PRIMARY KEY, client_ref TEXT UNIQUE, "
                "ts TEXT, data TEXT)")
    return con


_lock = threading.Lock()


def _field_name(nonce: str, fid: str) -> str:
    return "f_" + hmac.new(_SECRET, f"{nonce}:{fid}".encode(), hashlib.sha256).hexdigest()[:10]


def _token(nonce: str) -> str:
    return nonce + "." + hmac.new(_SECRET, nonce.encode(), hashlib.sha256).hexdigest()[:24]


def _check_token(tok: str) -> str | None:
    nonce, _, sig = tok.partition(".")
    return nonce if nonce and hmac.compare_digest(_token(nonce), tok) else None


portal_router = APIRouter(prefix="/mock-carrier-portal", tags=["mock-carrier-portal"])

PAGE_CSS = """<style>
:root{--bg:#f4f6fb;--card:#fff;--line:#d9deea;--text:#1d2233;--muted:#5d6480;--accent:#2d5be3;--bad:#c62f45;--good:#1b8a5a}
@media (prefers-color-scheme:dark){:root{--bg:#10131d;--card:#181c2a;--line:#2b3147;--text:#e8ebf6;--muted:#9aa2c0;--accent:#7a9bff;--bad:#ff7a8c;--good:#4fd39b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:14px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin:22px 0 8px}
.row{display:grid;grid-template-columns:240px 1fr;gap:10px;align-items:center;margin:8px 0}
label{color:var(--muted)}input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:transparent;color:var(--text);font:inherit}
.checks label{display:inline-flex;gap:6px;margin-right:14px;color:var(--text)}.checks input{width:auto}
button{margin-top:18px;background:var(--accent);color:#fff;border:0;border-radius:10px;padding:10px 18px;font-weight:600;cursor:pointer}
#errors li{color:var(--bad)}#confirmation{color:var(--good);font-size:18px;font-weight:700}
@media(max-width:600px){.row{grid-template-columns:1fr}}</style>"""


@portal_router.get("", response_class=HTMLResponse)
def portal_form() -> str:
    nonce = uuid.uuid4().hex[:16]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Carrier Portal</title>{PAGE_CSS}</head><body><main><div class='card'>"
             "<h1>Mock Carrier - Small Commercial Quote Request</h1><p style='color:var(--muted)'>ACORD 125 / 126 intake (test system)</p>"
             f"<form method='post' action='/mock-carrier-portal/submit' id='quote'>"
             f"<input type='hidden' name='form_token' value='{_token(nonce)}'>"
             f"<input type='hidden' name='client_ref' value=''>"]
    for label, fid, typ, req, opts in PORTAL_FIELDS:
        if typ == "section":
            parts.append(f"<h2>{html.escape(label)}</h2>")
            continue
        name = _field_name(nonce, fid)
        r = " required" if req else ""
        if typ == "select":
            o = "".join(f"<option value='{html.escape(v)}'>{html.escape(t)}</option>" for v, t in opts)
            ctl = f"<select id='{name}' name='{name}'{r}><option value=''>Select...</option>{o}</select>"
        elif typ == "checkbox":
            ctl = "<div class='checks'>" + "".join(
                f"<label><input type='checkbox' name='{name}' value='{v}'>{html.escape(t)}</label>" for v, t in opts) + "</div>"
        else:
            ctl = f"<input id='{name}' name='{name}' type='{typ}'{r}>"
        parts.append(f"<div class='row'><label for='{name}'>{html.escape(label)}</label>{ctl}</div>")
    parts.append("<button type='submit'>Submit for quote</button></form></div></main></body></html>")
    return "".join(parts)


def _validate(v: dict[str, list[str]]) -> tuple[dict, list[tuple[str, str]]]:
    one = lambda k: (v.get(k) or [""])[0].strip()
    errs: list[tuple[str, str]] = []
    out: dict = {}
    for label, fid, typ, req, opts in PORTAL_FIELDS:
        if typ == "section":
            continue
        val = v.get(fid, []) if typ == "checkbox" else one(fid)
        if req and not val:
            errs.append((label, "required"))
            continue
        if not val:
            continue
        if typ == "number":
            if not re.fullmatch(r"\d{1,12}", val):
                errs.append((label, "must be a whole number"))
                continue
            val = int(val)
        if typ in ("select", "checkbox"):
            allowed = {o for o, _ in opts}
            if any(x not in allowed for x in (val if isinstance(val, list) else [val])):
                errs.append((label, "invalid option"))
                continue
        out[fid] = val
    if "zip" in out and not re.fullmatch(r"\d{5}", out["zip"]):
        errs.append(("ZIP", "must be 5 digits"))
    if "email" in out and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", out["email"], re.I):
        errs.append(("Contact E-mail", "invalid e-mail"))
    if "eff" in out:
        try:
            if date.fromisoformat(out["eff"]).year < 2020:
                errs.append(("Requested Effective Date", "too far in the past"))
        except ValueError:
            errs.append(("Requested Effective Date", "must be YYYY-MM-DD"))
    if "occ" in out and "agg" in out and int(out["agg"]) < int(out["occ"]):
        errs.append(("General Aggregate Limit ($)", "must be >= each occurrence"))
    if "fein" in out and not re.fullmatch(r"\d{2}-\d{7}", out["fein"]):
        errs.append(("FEIN / Tax ID", "format NN-NNNNNNN"))
    return out, errs


@portal_router.post("/submit", response_class=HTMLResponse)
async def portal_submit(request: Request) -> HTMLResponse:
    pairs = parse_qsl((await request.body()).decode(), keep_blank_values=True)
    raw: dict[str, list[str]] = {}
    for k, val in pairs:
        raw.setdefault(k, []).append(val)
    nonce = _check_token((raw.get("form_token") or [""])[0])
    if not nonce:
        return HTMLResponse("<ul id='errors'><li data-field='form_token'>invalid or missing form token</li></ul>", 403)
    by_fid = {fid: raw.get(_field_name(nonce, fid), []) for _, fid, typ, _, _ in PORTAL_FIELDS if fid}
    data, errs = _validate(by_fid)
    if errs:
        items = "".join(f"<li data-field='{html.escape(l)}'>{html.escape(l)}: {html.escape(m)}</li>" for l, m in errs)
        return HTMLResponse(f"<!doctype html><title>Errors</title>{PAGE_CSS}<main><div class='card'><h1>Please fix</h1>"
                            f"<ul id='errors'>{items}</ul></div></main>", 422)
    ref = (raw.get("client_ref") or [""])[0] or None
    with _lock:
        con = _db()
        try:
            if ref and (row := con.execute("SELECT submission_id FROM submissions WHERE client_ref=?", [ref]).fetchone()):
                sid, dup = row[0], True
            else:
                sid, dup = "CP-" + uuid.uuid4().hex[:8].upper(), False
                con.execute("INSERT INTO submissions VALUES (?, ?, ?, ?)",
                            [sid, ref, datetime.now(timezone.utc).isoformat(), json.dumps(data)])
        finally:
            con.close()
    return HTMLResponse(f"<!doctype html><title>Submitted</title>{PAGE_CSS}<main><div class='card'>"
                        f"<div id='confirmation' data-submission-id='{sid}' data-duplicate='{str(dup).lower()}'>"
                        f"Submission received: {sid}</div></div></main>", 200)


def portal_app() -> FastAPI:
    app = FastAPI(title="Mock Carrier Portal")
    app.include_router(portal_router)
    return app


def get_submission(submission_id: str) -> dict | None:
    con = _db()
    try:
        row = con.execute("SELECT data, client_ref FROM submissions WHERE submission_id=?", [submission_id]).fetchone()
    finally:
        con.close()
    return {**json.loads(row[0]), "_client_ref": row[1]} if row else None


# ============================================================================ automation client
SYNONYMS = {
    "applicant_name": {"named", "insured", "applicant"}, "entity_type": {"legal", "entity"}, "fein": {"fein", "tax"},
    "mailing_street": {"street", "address"}, "mailing_city": {"city"}, "mailing_state": {"state"}, "mailing_zip": {"zip"},
    "business_phone": {"business", "phone"}, "contact_name": {"contact", "person"}, "contact_email": {"contact", "e-mail", "email"},
    "years_in_business": {"years", "business"}, "business_description": {"description", "operations"}, "naics": {"naics"},
    "employees_full_time": {"full-time", "employees"}, "employees_part_time": {"part-time", "employees"},
    "annual_revenue": {"annual", "gross", "sales"}, "policy_effective_date": {"effective", "date"},
    "lines_of_business": {"lines", "coverage"}, "gl_each_occurrence": {"each", "occurrence"},
    "gl_general_aggregate": {"general", "aggregate"}, "gl_deductible": {"deductible"}, "prior_losses_3y": {"losses"},
    "prior_carrier": {"carrier"},
}


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9-]+", s.lower()) if t and t not in {"of", "the", "in", "yrs", "past", "3", "id", "current", "prior", "requested"}}


class _FormParser(HTMLParser):
    """Collects <label for=...> texts, controls (input/select/checkbox groups) and hidden inputs."""

    def __init__(self):
        super().__init__()
        self.labels: dict[str, str] = {}
        self.controls: dict[str, dict] = {}
        self.hidden: dict[str, str] = {}
        self._in_label = False
        self._label_for: str | None = None
        self._text = ""
        self._row_label = ""          # text of the most recent <label for=...> (labels a checkbox group)
        self._select: str | None = None
        self._in_option = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "label":
            self._in_label, self._label_for, self._text = True, a.get("for"), ""
        elif tag == "input":
            t = a.get("type", "text")
            if t == "hidden":
                self.hidden[a.get("name", "")] = a.get("value", "")
            elif t == "checkbox":
                c = self.controls.setdefault(a["name"], {"type": "checkbox", "options": [], "label": self._row_label,
                                                         "required": False})
                c["options"].append((a.get("value", ""), ""))
            else:
                self.controls[a["name"]] = {"type": t, "required": "required" in a, "id": a.get("id")}
        elif tag == "select":
            self._select = a["name"]
            self.controls[a["name"]] = {"type": "select", "options": [], "required": "required" in a, "id": a.get("id")}
        elif tag == "option" and self._select:
            self._in_option = True
            self.controls[self._select]["options"].append((a.get("value", ""), ""))

    def handle_endtag(self, tag):
        if tag == "label" and self._in_label:
            if self._label_for:  # a row label; option labels inside checkbox groups carry no "for"
                self.labels[self._label_for] = self._text.strip()
                self._row_label = self._text.strip()
            self._in_label, self._label_for = False, None
        elif tag == "select":
            self._select = None
        elif tag == "option":
            self._in_option = False

    def handle_data(self, data):
        if self._in_label:
            self._text += data
        if self._in_option and self._select:
            v, t = self.controls[self._select]["options"][-1]
            self.controls[self._select]["options"][-1] = (v, (t + data).strip())


@dataclass
class FillResult:
    ok: bool
    submission_id: str | None
    status_code: int
    errors: list[str] = field(default_factory=list)
    mapping: dict[str, str] = field(default_factory=dict)   # portal label -> ACORD key
    unmapped_required: list[str] = field(default_factory=list)  # required portal fields left empty
    duplicate: bool = False


class InProcessTransport:
    """Talks to a FastAPI app without a network hop (Starlette TestClient)."""

    def __init__(self, app: FastAPI | None = None):
        from fastapi.testclient import TestClient

        self.client = TestClient(app or portal_app())

    def get(self, path: str) -> tuple[int, str]:
        r = self.client.get(path)
        return r.status_code, r.text

    def post_form(self, path: str, body: str) -> tuple[int, str]:
        r = self.client.post(path, content=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        return r.status_code, r.text


class HttpTransport:
    def __init__(self, base_url: str):
        import httpx

        self.c = httpx.Client(base_url=base_url, timeout=20)

    def get(self, path: str) -> tuple[int, str]:
        r = self.c.get(path)
        return r.status_code, r.text

    def post_form(self, path: str, body: str) -> tuple[int, str]:
        r = self.c.post(path, content=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        return r.status_code, r.text


class PortalClient:
    def __init__(self, transport=None, path: str = "/mock-carrier-portal"):
        self.t = transport or InProcessTransport()
        self.path = path

    @staticmethod
    def _map(label: str) -> str | None:
        toks = _tokens(label)
        best, score = None, 0.0
        for key, syn in SYNONYMS.items():
            s = len(toks & syn) / len(toks | syn) if toks | syn else 0
            if s > score:
                best, score = key, s
        return best if score >= 0.3 else None

    @staticmethod
    def _fmt(value, ctl: dict) -> list[str]:
        if value is None:
            return []
        typ = ctl["type"]
        if typ == "checkbox":
            vals = value if isinstance(value, list) else [value]
            allowed = {o for o, _ in ctl["options"]}
            return [v for v in vals if v in allowed]
        if typ == "select":
            want = str(value).lower()
            for v, t in ctl["options"]:
                if v and (v.lower() == want or t.lower() == want or re.sub(r"[^0-9]", "", t) == str(value)):
                    return [v]
            return []
        if typ == "number":
            return [str(int(value))]
        if typ == "tel" and isinstance(value, str) and value.startswith("+1"):
            return [f"({value[2:5]}) {value[5:8]}-{value[8:]}"]
        return [str(value)]

    def fill(self, record: dict, client_ref: str) -> FillResult:
        code, page = self.t.get(self.path)
        if code != 200:
            return FillResult(False, None, code, [f"GET form failed: {code}"])
        p = _FormParser()
        p.feed(page)
        id_to_name = {c.get("id"): n for n, c in p.controls.items() if c.get("id")}
        pairs: list[tuple[str, str]] = [("form_token", p.hidden.get("form_token", "")), ("client_ref", client_ref)]
        mapping, unmapped = {}, []
        for name, ctl in p.controls.items():
            label = p.labels.get(ctl.get("id") or "", "") if ctl["type"] != "checkbox" else ctl.get("label", "")
            key = self._map(label)
            values = self._fmt(record.get(key), ctl) if key else []
            if key:
                mapping[label] = key
                pairs.extend((name, v) for v in values)
            if ctl.get("required") and not values:
                unmapped.append(label if key else f"{label} (no matching ACORD field)")
        code, resp = self.t.post_form(self.path + "/submit", urlencode(pairs))
        m = re.search(r"data-submission-id='([^']+)' data-duplicate='(\w+)'", resp)
        errors = re.findall(r"<li data-field='[^']*'>([^<]+)</li>", resp)
        return FillResult(bool(m) and code == 200, m.group(1) if m else None, code, [html.unescape(e) for e in errors],
                          mapping, unmapped, bool(m and m.group(2) == "true"))
