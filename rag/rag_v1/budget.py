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

