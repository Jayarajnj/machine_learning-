# %% [markdown]
# # Multi-DB research chat: neighbour / thread context expansion (step 7)
# A retrieved chunk read alone can mislead (a reply without the message it answers, a table row
# without its heading). After retrieval, each hit is shown together with the chunks around it, or
# its whole email thread, in order. Expansion is READ-ONLY and EPHEMERAL: it builds the text shown
# to the model each turn; the stored working set keeps the plain hits.
# Hits are grouped per DB by a metadata "group" (thread / document) and ordered by metadata keys, so
# it works with whatever fields your DBs end up having; see `ExpansionSpec`.
# Needs research_retrieval_core.py in the same folder.

# %%
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

from research_retrieval_core import Chunk, ChromaRetriever, DBRegistry

log = logging.getLogger("research_context")


# %% [markdown]
# ## Cell 1: how chunks are grouped and ordered (configurable, auto-detected from common names)

# %%
DEFAULT_GROUP_KEYS = ("thread_id", "thread", "conversation_id", "parent_id", "email_id",
                      "doc_id", "document_id", "source", "filename", "file_name", "file", "path")
DEFAULT_ORDER_KEYS = ("date", "timestamp", "sent_at", "message_index", "chunk_index",
                      "chunk", "position", "seq", "index")          # composite sort: all present keys, in this order
DEFAULT_INDEX_KEYS = ("chunk_index", "chunk", "message_index", "position", "seq", "index")  # consecutive ints

def _blank(v) -> bool:
    return v is None or v == ""

def _sortable(v):
    """Comparable across mixed types (numbers < text < missing)."""
    if _blank(v):
        return (2, 0, "")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (0, v, "")
    return (1, 0, str(v))

def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)

@dataclass
class ExpansionSpec:
    enabled: bool = True
    group_keys: tuple = DEFAULT_GROUP_KEYS     # first one present on a chunk defines its group (thread / document)
    order_keys: tuple = DEFAULT_ORDER_KEYS     # how chunks inside a group are ordered (all present keys, in order)
    index_keys: tuple = DEFAULT_INDEX_KEYS     # consecutive integer position, used for huge groups only
    mode: str = "window"                       # "window": +-`window` chunks | "thread": the whole group 

if small
    window: int = 1
    thread_max_chunks: int = 12                # "thread" mode: groups larger than this fall back to the window
    max_group_fetch: int = 400                 # groups bigger than this are expanded with a numeric range query
    max_context_chars: int = 2400              # cap per expanded passage (far neighbours are dropped first)

    def group_of(self, md: dict):
        for k in self.group_keys:
            if k in md and not _blank(md[k]):
                return (k, md[k])
        return None

    def has_order(self, md: dict) -> bool:
        return any(k in md and not _blank(md[k]) for k in self.order_keys)

    def sort_key(self, md: dict) -> tuple:
        return tuple(_sortable(md.get(k)) for k in self.order_keys)

    def index_key_of(self, md: dict) -> Optional[str]:
        return next((k for k in self.index_keys if _is_num(md.get(k))), None)

# %% [markdown]
# ## Cell 2: the expander
# Per DB: group the hits, load each group's chunks, order them, select the window (or thread) around
# every hit, merge overlapping windows into one passage, and attach it to the best-ranked hit of that
# passage (`chunk.context`, joined with " ¶ "). Other hits inside the same passage are absorbed, so a
# passage is cited once. Any failure leaves the plain hits untouched.

# %%
SEP = " ¶ "


class ContextExpander:
    def __init__(self, registry: DBRegistry, specs: Optional[dict] = None,
                 default: Optional[ExpansionSpec] = None, enabled: bool = True):
        self.registry, self.specs = registry, specs or {}
        self.default, self.enabled = default or ExpansionSpec(), enabled
        self._warned: set = set()

    def expand(self, grouped: dict) -> dict:
        """{db: [Chunk]} -> {db: [Chunk]} where hits carry `context`. Never mutates the input chunks."""
        if not self.enabled:
            return grouped
        out = {}
        for db, hits in grouped.items():
            spec = self.specs.get(db, self.default)
            if not spec.enabled or not hits:
                out[db] = list(hits)
                continue
            try:
                out[db] = self._expand_db(self.registry.get(db), list(hits), spec)
            except Exception as e:                                     # noqa: BLE001
                log.warning("Context expansion failed for DB %r: %s", db, e)
                out[db] = list(hits)
        return out

    # -- loading a group's chunks -------------------------------------------------------
    def _load_group(self, retr: ChromaRetriever, gk, gv, members: list, spec: ExpansionSpec):
        """Returns (chunks sorted in order, gap_key or None) or None if the group can't be loaded."""
        got = retr.get_where({gk: gv}, limit=spec.max_group_fetch + 1)
        if len(got) <= spec.max_group_fetch:
            return sorted(got, key=lambda c: spec.sort_key(c.metadata)), None
        # huge group: fetch only +-window around each hit using a consecutive integer position key
        ik = spec.index_key_of(members[0][1].metadata)
        if ik is None:
            log.warning("Group %s=%r has more than %d chunks and no integer position key: not expanded",
                        gk, gv, spec.max_group_fetch)
            return None
        by_id = {}
        for _, h in members:
            pos = h.metadata.get(ik)
            if not _is_num(pos):
                continue
            where = {"$and": [{gk: gv}, {ik: {"$gte": pos - spec.window}}, {ik: {"$lte": pos + spec.window}}]}
            for c in retr.get_where(where):
                by_id[c.id] = c
        return sorted(by_id.values(), key=lambda c: spec.sort_key(c.metadata)), ik


    @staticmethod
    def _trim(run: list, hit_pos: set, chunks: list, cap: int) -> list:
        """Drop far-away neighbours from the ends of a run until the passage fits `cap` characters."""
        run = list(run)

        def size(r):
            return sum(len(chunks[j].text) for j in r) + len(SEP) * (len(r) - 1)

        while size(run) > cap and len(run) > 1:
            ends = [e for e in (run[0], run[-1]) if e not in hit_pos]
            if not ends:
                break
            run.remove(max(ends, key=lambda e: min(abs(e - p) for p in hit_pos)))
        return run

    # -- one DB -----------------------------------------------------------------------------
    def _expand_db(self, retr: ChromaRetriever, hits: list, spec: ExpansionSpec) -> list:
        result: dict = {}                       # rank -> chunk to show
        groups: dict = {}                       # (key, value) -> [(rank, hit)]
        for rank, h in enumerate(hits):
            g = spec.group_of(h.metadata)
            if g is None or not spec.has_order(h.metadata):
                result[rank] = h                # no grouping/ordering metadata: leave as is
            else:
                groups.setdefault(g, []).append((rank, h))

        if not groups and hits and retr.name not in self._warned:
            self._warned.add(retr.name)
            log.warning("DB %r: no hit has both a group field (%s...) and an order field (%s...) in its metadata, so "
                        "context expansion does nothing for it. Store e.g. thread_id/source and chunk_index.",
                        retr.name, spec.group_keys[0], spec.order_keys[0])
        for (gk, gv), members in groups.items():
            loaded = self._load_group(retr, gk, gv, members, spec)
            if loaded is None:
                result.update({rank: h for rank, h in members})
                continue
            chunks, gap_key = loaded
            pos_by_id = {c.id: i for i, c in enumerate(chunks)}
            hit_pos: dict = {}
            for rank, h in members:
                if h.id in pos_by_id:
                    hit_pos[rank] = pos_by_id[h.id]
                else:
                    result[rank] = h

            if spec.mode == "thread" and len(chunks) <= spec.thread_max_chunks:
                selected = set(range(len(chunks)))
            else:
                selected = {j for p in hit_pos.values()
                            for j in range(max(0, p - spec.window), min(len(chunks), p + spec.window + 1))}

            runs: list = []
            for j in sorted(selected):
                contiguous = bool(runs) and j == runs[-1][-1] + 1
                if contiguous and gap_key is not None:      # huge-group mode: positions may skip numbers
                    a, b = chunks[runs[-1][-1]].metadata.get(gap_key), chunks[j].metadata.get(gap_key)
                    contiguous = _is_num(a) and _is_num(b) and b - a <= 1
                if contiguous:
                    runs[-1].append(j)
                else:
                    runs.append([j])

            for run in runs:
                in_run = {rank: p for rank, p in hit_pos.items() if run[0] <= p <= run[-1]}
                if not in_run:
                    continue
                best_rank = max(in_run, key=lambda r: (hits[r].scores.get("rerank", 0.0), -r))
                run = self._trim(run, set(in_run.values()), chunks, spec.max_context_chars)
                if len(run) == 1:
                    result[best_rank] = hits[best_rank]     # no neighbours available
                    continue
                result[best_rank] = replace(
                    hits[best_rank],
                    context=SEP.join(chunks[j].text.strip() for j in run),
                    context_ids=[chunks[j].id for j in run])
                # the other hits inside this passage are absorbed into it
        return [result[r] for r in sorted(result)]


# %% [markdown]
# ## Cell 3: offline smoke test

# %%
def _smoke_test():
    from research_retrieval_core import _FakeCollection

    def thread(tid, texts, start=0):
        return [(f"{tid}-{i}", t, {"thread_id": tid, "chunk_index": i, "source": f"mail_{tid}"})
                for i, t in enumerate(texts, start)]

    t1 = [f"T1 part {i}: " + "x" * 70 for i in range(5)]
    rows = thread("T-1", t1) + thread("T-2", ["T2 part 0", "T2 part 1"]) + [("loose", "no metadata at all", {})]
    coll = _FakeCollection(rows)
    reg = DBRegistry().add(ChromaRetriever("mails", coll, description="Email threads"))

    def hit(cid, score=1.0, db="mails"):
        r = next(r for r in rows if r[0] == cid)
        return Chunk(id=cid, text=r[1], db=db, metadata=dict(r[2]), source=f"src-{cid}", scores={"rerank": score})

    ex = ContextExpander(reg)

    # 1) window=1: the hit comes with its neighbours, in order; originals untouched
    h2 = hit("T-1-2")
    out = ex.expand({"mails": [h2]})["mails"]
    assert len(out) == 1 and out[0].id == "T-1-2" and out[0].context_ids == ["T-1-1", "T-1-2", "T-1-3"]
    assert out[0].context == SEP.join(t1[1:4]) and out[0].full_text == out[0].context and out[0].text == t1[2]
    assert h2.context == "" and h2.full_text == h2.text


    # 2) two close hits merge into ONE passage attributed to the better-ranked hit
    out = ex.expand({"mails": [hit("T-1-1", 0.6), hit("T-1-3", 0.9)]})["mails"]
    assert len(out) == 1 and out[0].id == "T-1-3" and out[0].context_ids == [f"T-1-{i}" for i in range(5)]


    # 3) far-apart hits stay separate passages (ranks preserved)
    out = ex.expand({"mails": [hit("T-1-4", 0.9), hit("T-1-0", 0.8)]})["mails"]
    assert [c.id for c in out] == ["T-1-4", "T-1-0"]
    assert out[0].context_ids == ["T-1-3", "T-1-4"] and out[1].context_ids == ["T-1-0", "T-1-1"]

    # 4) chunks without group/order metadata are returned unchanged, and order is kept
    out = ex.expand({"mails": [hit("loose"), hit("T-1-2")]})["mails"]
    assert out[0].context == "" and out[1].context_ids == ["T-1-1", "T-1-2", "T-1-3"]

    # 5) thread mode: the whole (small) thread, but not other threads
    ex_thread = ContextExpander(reg, default=ExpansionSpec(mode="thread"))
    out = ex_thread.expand({"mails": [hit("T-1-2")]})["mails"]
    assert out[0].context_ids == [f"T-1-{i}" for i in range(5)] and "T2 part" not in out[0].context
    ex_big = ContextExpander(reg, default=ExpansionSpec(mode="thread", thread_max_chunks=3))   # too big -> window
    assert ex_big.expand({"mails": [hit("T-1-2")]})["mails"][0].context_ids == ["T-1-1", "T-1-2", "T-1-3"]

    # 6) character cap: far neighbours are dropped first, the hit stays
    ex_cap = ContextExpander(reg, default=ExpansionSpec(window=2, max_context_chars=260))
    out = ex_cap.expand({"mails": [hit("T-1-2")]})["mails"]
    assert out[0].context_ids == ["T-1-1", "T-1-2", "T-1-3"]

    # 7) huge group: falls back to numeric range queries and keeps distant windows apart
    ex_huge = ContextExpander(reg, default=ExpansionSpec(max_group_fetch=3))
    out = ex_huge.expand({"mails": [hit("T-1-0", 0.9), hit("T-1-4", 0.8)]})["mails"]
    assert [c.context_ids for c in out] == [["T-1-0", "T-1-1"], ["T-1-3", "T-1-4"]]

    # 8) per-DB switch, global switch, and failure safety
    assert ContextExpander(reg, specs={"mails": ExpansionSpec(enabled=False)}).expand({"mails": [hit("T-1-2")]})["mails"][0].context == ""
    assert ContextExpander(reg, enabled=False).expand({"mails": [hit("T-1-2")]})["mails"][0].context == ""
    retr = reg.get("mails")
    original = retr.get_where
    retr.get_where = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))
    assert ContextExpander(reg).expand({"mails": [hit("T-1-2")]})["mails"][0].context == ""
    retr.get_where = original
    # a DB without grouping/ordering metadata is left alone, and the reason is logged once
    import logging

    class _Catch(logging.Handler):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def emit(self, record):
            self.msgs.append(record.getMessage())
    catch = _Catch()
    log.addHandler(catch)
    bare = ContextExpander(DBRegistry().add(ChromaRetriever("bare", _FakeCollection([("z", "text", None)]))))
