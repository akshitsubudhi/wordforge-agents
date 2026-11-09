"""
================================================================================
BLOCK: Curated Word List Builder (dev-time only; NOT a runtime dependency)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : a Hunspell en_US.dic (LibreOffice dictionaries) + the `wordfreq`
             package (Zipf frequencies). Both are needed ONLY to regenerate.
  Downstream: data/wordlist.tsv  (word \t zipf), committed to the repo. Loaded
             by core/wordlist.py for the game referee, the trie solver and the
             Phase-2 difficulty model (Zipf rarity feature).

Key engineering mechanism
  1. Take only entries whose Hunspell stem is all-lowercase. Hunspell spells
     proper nouns ("John/SM", "Paris/M") and abbreviations ("USA/M") with
     capitals, so this one rule drops names and acronyms without a hand list.
  2. Expand the plural (S), past (D) and gerund (G) affix flags with standard
     English spelling rules. A mis-generated form ("bakeing") has a Zipf
     frequency of ~0, so the frequency floor below discards it.
  3. Keep 3..7 letter, vowel-bearing a-z tokens, drop a profanity/sensitive
     blocklist, rank by Zipf and keep the top N (default 2,500).

Failure modes & 1-step mitigation
  * `pip install wordfreq` hangs (seen in the sandbox) -> use
    `uv pip install wordfreq` (works) or simply don't rebuild: the TSV is
    committed, so runtime never imports wordfreq.
  * Dictionary drift between regenerations -> output is sorted + deterministic;
    `make wordlist` diff shows exactly which words changed.
================================================================================
Usage:
  uv pip install wordfreq
  curl -L -o /tmp/en_US.dic https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic
  python scripts/build_wordlist.py --dic /tmp/en_US.dic --out data/wordlist.tsv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VOWELS = set("aeiouy")
ALPHA = re.compile(r"^[a-z]+$")

# Offensive / sensitive tokens that are valid English but unwanted in a
# family word game. Kept intentionally short and explicit.
BLOCKLIST = {
    "ass", "asses", "damn", "damned", "hell", "sex", "sexy", "porn", "nude",
    "naked", "kill", "killed", "killing", "kills", "rape", "raped", "slut",
    "whore", "bitch", "crap", "piss", "pissed", "fuck", "shit", "dick",
    "cock", "tit", "tits", "nazi", "drug", "drugs", "gun", "guns", "suicide",
    "murder", "dead", "die", "died", "dies", "dying", "bomb", "terror",
    "sexual", "gay", "fat", "god", "jesus", "lord", "bible", "church",
    "christ", "muslim", "jewish", "islam", "racist", "abortion", "breast",
    "breasts", "butt", "drunk", "beer", "wine", "vodka", "whiskey", "bloody",
    "weed", "pee", "poop", "boob", "boobs", "horny", "lust", "kinky", "pimp",
    # interjections / fragments / abbreviations that slip through as lowercase
    "etc", "aka", "yeah", "yep", "nope", "okay", "hey", "huh", "wow", "ugh",
    "lol", "omg", "hmm", "mmm", "oops", "yay", "aww", "ooh", "hah", "haha",
    "gonna", "wanna", "gotta", "kinda", "sorta", "dude", "bro", "cos", "mrs",
    "ins", "ive", "dont", "cant", "wont", "isnt", "doesnt", "didnt", "thats",
    # proper-noun / brand / demonym dominated tokens (frequency comes from the
    # capitalised sense, even though a rare lowercase sense exists)
    "amazon", "billy", "bob", "carter", "charlie", "china", "dutch", "french",
    "google", "harry", "inc", "japan", "jimmy", "john", "johnny", "josh",
    "lee", "mac", "maria", "martin", "max", "mike", "min", "oxford", "peter",
    "roger", "roman", "soviet", "tom", "tony", "trans", "trump", "twitter",
    "ups", "wales", "gods", "asshole", "facebook", "obama", "covid", "uber",
    # profanity inflections, numerals, abbreviations, bare prefixes, harsh words
    "fucked", "fucking", "suck", "sucks", "iii", "vol", "tho", "nah", "rep",
    "non", "anti", "multi", "semi", "virgin", "killer", "assault", "idiot",
}


def expand(stem: str, flags: str) -> set[str]:
    """Generate inflected forms for the Hunspell affix flags we care about."""
    forms = {stem}
    cons_y = len(stem) > 1 and stem.endswith("y") and stem[-2] not in "aeiou"
    if "S" in flags:  # plural / 3rd person
        if cons_y:
            forms.add(stem[:-1] + "ies")
        elif stem.endswith(("s", "x", "z", "ch", "sh")):
            forms.add(stem + "es")
        else:
            forms.add(stem + "s")
    if "D" in flags:  # past tense
        if stem.endswith("e"):
            forms.add(stem + "d")
        elif cons_y:
            forms.add(stem[:-1] + "ied")
        else:
            forms.add(stem + "ed")
    if "G" in flags:  # gerund
        if stem.endswith("e") and not stem.endswith("ee"):
            forms.add(stem[:-1] + "ing")
        else:
            forms.add(stem + "ing")
    return forms


def build(dic_path: Path, size: int, min_len: int, max_len: int) -> list[tuple[str, float]]:
    try:
        from wordfreq import zipf_frequency
    except ImportError:  # pragma: no cover - dev-only path
        sys.exit("wordfreq missing. Install with: uv pip install wordfreq")

    candidates: set[str] = set()
    lines = dic_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]
    for line in lines:
        stem, _, flags = line.strip().partition("/")
        if not stem or not ALPHA.match(stem):  # drops Capitalised names + ACRONYMS
            continue
        candidates |= expand(stem, flags)

    scored = []
    for w in candidates:
        if not (min_len <= len(w) <= max_len) or not ALPHA.match(w):
            continue
        if not (set(w) & VOWELS) or w in BLOCKLIST:
            continue
        z = zipf_frequency(w, "en")
        if z >= 2.5:
            scored.append((w, round(z, 2)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return sorted(scored[:size])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    ap.add_argument("--dic", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/wordlist.tsv"))
    ap.add_argument("--size", type=int, default=2500)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=7)
    a = ap.parse_args()
    rows = build(a.dic, a.size, a.min_len, a.max_len)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as fh:
        fh.write("# word\tzipf  (generated by scripts/build_wordlist.py; Hunspell en_US lowercase stems x wordfreq)\n")
        for w, z in rows:
            fh.write(f"{w}\t{z}\n")
    print(f"wrote {len(rows)} words -> {a.out}  (zipf range {min(z for _, z in rows)}..{max(z for _, z in rows)})")


if __name__ == "__main__":
    main()
