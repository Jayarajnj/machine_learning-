# %% [markdown]
# # Multi-DB research chat: near-duplicate removal (step 9)
# The same text often lives in more than one place (an email attachment that is also a policy document,
# a quoted reply, two copies of a PDF). Without this, both copies fill the top-k and the prompt.
# `Deduper` collapses near-identical chunks, within a DB and across DBs, keeping the best one and noting
# where else the text appears (`chunk.also_in`), so provenance (e.g. the email that carried it) is not lost.
#
# Deliberately conservative: two chunks are only "the same" when
#   * their word 3-gram overlap is high (Jaccard >= 0.8) or one is almost entirely inside the other, AND
#   * they contain the same numbers (a policy that says "3 days" is NOT a duplicate of one that says "2 days"), AND
#   * their version/revision metadata does not differ, AND
#   * both are long enough (short lines such as "Approved." or signatures are never merged).
# Text-based on purpose: embeddings from different DBs/models are not comparable. Needs research_retrieval_core.

# %%
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Optional

from research_retrieval_core import Chunk

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list:
    return _TOKEN.findall((text or "").lower())
@dataclass
class DedupeConfig:
    enabled: bool = True
    jaccard: float = 0.8              # word-3-gram overlap that counts as "the same text"
    containment: float = 0.9          # share of the smaller chunk found inside the larger one
    shingle: int = 3
    min_tokens: int = 12              # shorter chunks are never merged (boilerplate lines are not duplicates)
    score_tolerance: float = 0.05     # rerank scores this close count as equal when choosing which copy to keep
    db_priority: tuple = ()           # most authoritative DB first, e.g. ("policies", "emails"): wins ties
    guard_keys: tuple = ("version", "revision", "doc_version", "effective_date")   # differing values = not a duplicate
    cross_db: bool = True
    within_db: bool = True

@dataclass
class _Entry:
    chunk: Chunk
    pos: tuple                        # (db index, rank in its DB): keeps the original order stable
    shingles: Optional[frozenset]
    nums: frozenset
    also_in: list
class Deduper:
    def __init__(self, config: Optional[DedupeConfig] = None):
        self.cfg = config or DedupeConfig()

    # -- features ------------------------------------------------------------------
    def _entry(self, chunk: Chunk, pos: tuple) -> _Entry:
        toks = _tokens(chunk.text)
        n = self.cfg.shingle
        sh = None
        if len(toks) >= self.cfg.min_tokens:
            sh = frozenset(tuple(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1)))
        nums = frozenset(t for t in toks if any(ch.isdigit() for ch in t))
        return _Entry(chunk, pos, sh, nums, list(chunk.also_in))
    @staticmethod
    def _label(c: Chunk) -> str:
        return f"{c.db}: {c.source}"
