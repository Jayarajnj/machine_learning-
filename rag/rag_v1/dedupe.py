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

