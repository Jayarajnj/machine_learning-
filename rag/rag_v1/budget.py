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
