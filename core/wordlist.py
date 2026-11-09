"""
================================================================================
BLOCK: Curated Lexicon (dictionary + Zipf rarity)
--------------------------------------------------------------------------------
Role & contracts
  Upstream : data/wordlist.tsv (built by scripts/build_wordlist.py, committed).
  Downstream: game referee (membership), trie solver (enumeration), Phase-2
             difficulty model (zipf(word) rarity feature).
  Contract : words are lowercase a-z, 3..7 letters; zipf in [~4.3, ~7.8].

Key engineering mechanism
  Loaded once, memoised (lru_cache); O(1) membership via frozenset; Zipf map
  for difficulty features. Fully offline - no wordfreq import at runtime.

Failure modes & 1-step mitigation
  * TSV missing/corrupt -> FileNotFoundError with the exact rebuild command.
  * Names/abbreviations leaking into play -> list is filtered at build time
    (Hunspell lowercase stems + curated blocklist), not at request time.
================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from core.config import get_settings


@dataclass(frozen=True)
class Lexicon:
    words: frozenset[str]
    zipf: dict[str, float]

    def __contains__(self, w: str) -> bool:
        return w.lower() in self.words

    def rarity(self, w: str) -> float:
        """Higher = rarer. 8 - zipf keeps the feature positive and monotone."""
        return round(8.0 - self.zipf.get(w.lower(), 3.0), 3)


@lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> Lexicon:
    p = Path(path) if path else get_settings().wordlist_path
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Rebuild with: uv pip install wordfreq && "
            "python scripts/build_wordlist.py --dic en_US.dic"
        )
    zipf: dict[str, float] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        w, _, z = line.partition("\t")
        zipf[w.strip()] = float(z or 0)
    return Lexicon(words=frozenset(zipf), zipf=zipf)
