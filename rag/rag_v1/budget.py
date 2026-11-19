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

