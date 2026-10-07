"""Document-level decontamination: exact, prefix and near-duplicate matches against test corpora / positives."""
from olmo3_embed.data.doc_decontam import DocIndex

ARG = ("Freedom of expression should not be suppressed by accusations of sexism. Firstly, I would like to point out "
       "that the motion assumes every critic acts in bad faith, which is not supported by any of the evidence offered "
       "in the opening speech, and secondly the examples given are all drawn from a single online forum.")
WIKI = ("Earl of Hardwicke is a title in the Peerage of Great Britain. It was created in 1754 for the lawyer and "
        "politician Philip Yorke, 1st Baron Hardwicke, who served as Lord Chancellor from 1737 to 1756 and was a "
        "prominent figure in the Whig administrations of the mid eighteenth century.")


def idx():
    d = DocIndex()
    d.add_full("Touche2020", ARG)
    d.add_positive("DBPedia", "dbp:1", WIKI)
    return d


def test_exact_and_normalization():
    d = idx()
    assert d.check(ARG)[:2] == ("doc_exact", "Touche2020")
    assert d.check("  " + ARG.upper().replace(",", " ,") + " ")[:2] == ("doc_exact", "Touche2020")  # case/punct/space


def test_prefix_catches_truncated_or_extended_copies():
    d = idx()
    assert d.check(ARG + " Thirdly, more text that the test copy does not have.")[:2] == ("doc_prefix", "Touche2020")


def test_positive_exact_and_near_duplicate():
    d = idx()
    assert d.check(WIKI)[:2] == ("pos_exact", "DBPedia")
    near = WIKI.replace("prominent", "leading")  # one-word edit of a copied passage (Jaccard ~0.89)
    assert d.check(near)[:2] == ("pos_near", "DBPedia")
    rewritten = WIKI.replace("lawyer and politician", "jurist").replace("prominent", "leading").replace("Whig", "Tory")
    assert d.check(rewritten) is None  # several edits: Jaccard < 0.8, treated as a different passage


def test_clean_text_passes():
    d = idx()
    assert d.check("The mitochondrion is the powerhouse of the cell, producing ATP through oxidative phosphorylation.") is None
    assert d.check("") is None


def test_long_positive_exact_via_full_hash():
    from olmo3_embed.data.decontam import h
    long_doc = WIKI + " " + ARG  # > 300 chars: only a prefix is cached, the exact match uses the full-text hash
    d = DocIndex()
    d.add_positive("SCIDOCS", "s:9", long_doc[:300], full_hash=h(long_doc))
    assert d.check(long_doc)[:2] == ("pos_exact", "SCIDOCS")
