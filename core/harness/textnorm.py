"""
================================================================================
BLOCK: Text Normalisation & Obfuscation Codecs (shared by Layer-2 and the mock)
--------------------------------------------------------------------------------
Role & contracts
  Pure functions, no I/O. Used by:
    * redteam_guard.UntrustedContentSanitizer (DEFENDER: what we neutralise)
    * llm.MockLLM(mode="naive") (ATTACKER MODEL: what an LLM can *read*)
  Keeping both on one module makes the threat model explicit: the mock reads
  strictly MORE encodings (ROT13, letter-spacing, Unicode tag smuggling) than
  Layer-2 neutralises - which is exactly why Layer-2 alone cannot be enough.

Key engineering mechanism
  * NFKC folds full-width / mathematical / ligature look-alikes.
  * Invisible stripper: zero-width, bidi overrides, soft hyphen, BOM and the
    Unicode TAG block (U+E0000-E007F "ASCII smuggling").
  * Mixed-script confusable mapping (UTS#39 style): only tokens that MIX Latin
    with Cyrillic/Greek are folded, so legitimate all-Cyrillic text (a Russian
    customer email) is never corrupted -> no false positives on i18n input.
  * Directive grammar `CALL <tool> {json}` parsed with json.raw_decode for
    exact span boundaries (balanced braces, escaped quotes).

Failure modes & 1-step mitigation
  * Exotic confusables outside the table -> NFKC catches most; residual risk
    is covered by Layer-1 argument rules + Layer-3 approvals (defence in depth).
  * Huge inputs (regex CPU) -> callers truncate BEFORE normalising.
================================================================================
"""
from __future__ import annotations

import base64
import binascii
import codecs
import json
import re
import unicodedata
from functools import lru_cache

INVISIBLE_RE = re.compile(
    "[­᠎​-‏‪-‮⁠-⁤⁦-⁯﻿\U000e0000-\U000e007f]"
)
TAG_RE = re.compile("[\U000e0020-\U000e007e]")

CONFUSABLES = {
    # Cyrillic
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "У": "Y", "Ү": "Y", "Ѕ": "S", "І": "I", "Ј": "J", "Ԁ": "D", "Ԛ": "Q",
    "Ԝ": "W", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "ԁ": "d", "ԛ": "q", "ԝ": "w", "һ": "h", "ӏ": "l",
    # Greek
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N",
    "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "ο": "o", "ν": "v", "ι": "i", "κ": "k",
    "ρ": "p", "α": "a", "τ": "t", "υ": "u",
}
_CONF_TABLE = str.maketrans(CONFUSABLES)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DIRECTIVE_RE = re.compile(r"\bCALL\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?=\{)")
B64_RUN_RE = re.compile(r"[A-Za-z0-9+/_-]{24,}={0,2}")


@lru_cache(maxsize=4096)
def _script(ch: str) -> str:
    if not ch.isalpha():
        return ""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "OTHER"
    for s in ("LATIN", "CYRILLIC", "GREEK"):
        if name.startswith(s):
            return s
    return "OTHER"


def strip_invisible(s: str) -> tuple[str, int]:
    out, n = INVISIBLE_RE.subn("", s)
    return out, n


def decode_tags(s: str) -> str:
    """What a tokenizer 'sees' in Unicode-tag smuggling: map U+E00xx -> ASCII."""
    return TAG_RE.sub(lambda m: chr(ord(m.group()) - 0xE0000), s)


def map_confusables_mixed(s: str) -> tuple[str, int]:
    """Fold look-alikes only inside tokens that mix Latin with Cyrillic/Greek."""
    count = 0

    def fix(m: re.Match) -> str:
        nonlocal count
        tok = m.group()
        scripts = {_script(c) for c in tok} - {""}
        if "LATIN" in scripts and scripts & {"CYRILLIC", "GREEK"}:
            count += 1
            return tok.translate(_CONF_TABLE)
        if len(tok) >= 4 and scripts and scripts <= {"CYRILLIC", "GREEK"} and all(c in CONFUSABLES or not c.isalpha() for c in tok):
            count += 1  # whole-token disguise e.g. "САРТ"
            return tok.translate(_CONF_TABLE)
        return tok

    return TOKEN_RE.sub(fix, s), count


def normalize(s: str) -> tuple[str, dict]:
    s = unicodedata.normalize("NFKC", s)
    s, n_inv = strip_invisible(s)
    s, n_conf = map_confusables_mixed(s)
    return s, {"invisible_removed": n_inv, "confusable_tokens": n_conf}


def try_b64_text(blob: str) -> str | None:
    """Decode base64/base64url; return text only if it is mostly printable."""
    pad = "=" * (-len(blob) % 4)
    for dec in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            raw = dec(blob + pad)
        except (binascii.Error, ValueError):
            continue
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if txt and sum(c.isprintable() or c in "\n\t" for c in txt) / len(txt) > 0.9:
            return txt
    return None


def rot13(s: str) -> str:
    return codecs.encode(s, "rot13")


def unletterspace(s: str) -> str:
    """'C A L L   s e t' -> 'CALL set' (2+ spaces separate words)."""
    words = re.split(r" {2,}", s)
    return " ".join(w.replace(" ", "") if re.fullmatch(r"(\S )+\S", w) else w for w in words)


def find_directives(text: str) -> list[tuple[int, int, str, dict]]:
    """Return (start, end, tool, args) for every well-formed `CALL tool {json}`."""
    out = []
    dec = json.JSONDecoder()
    for m in DIRECTIVE_RE.finditer(text):
        try:
            obj, end = dec.raw_decode(text, m.end())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append((m.start(), end, m.group(1), obj))
    return out


def canonical(name: str, args: dict) -> str:
    return name + json.dumps(args, sort_keys=True, separators=(",", ":"))
