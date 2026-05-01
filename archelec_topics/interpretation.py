"""Interpretation and naming of topics produced by LDA / NMF.

Topic modelling is unsupervised and the course slides are explicit about the
fact that "topics require human interpretation". This module bundles the
helpers I use to do that interpretation rigorously:

  - browse the top-N words of each topic in a printable / serializable form
    (`topic_words_dataframe`, `make_label_template`),
  - find the few documents most representative of each topic, which is the
    second most useful tool (after the top words) to attach a human label
    (`find_representative_documents`),
  - apply manual labels via a {topic_id: name} dict and produce a list of
    display labels (`apply_labels`),
  - align LDA and NMF topics by lexical overlap so that the two models can be
    compared topic-by-topic (`align_topics`).

Heuristic / suggested labels are explicitly rejected here: in an academic
report, every topic name ends up being justified in prose, so an automatic
suggestion would only push us toward post-hoc rationalization of bad labels.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Topic content
# =============================================================================

def topic_words_dataframe(model, n_words=12, topic_labels=None):
    """Return a DataFrame with one column per topic and `n_words` rows.

    Parameters
    ----------
    model : TopicModelBase
        A fitted LDATopicModel or NMFTopicModel.
    n_words : int
        Number of top words to display per topic.
    topic_labels : dict[int, str] or list[str], optional
        If provided, used as column headers in place of "Topic k".

    Returns
    -------
    pd.DataFrame
        Shape (n_words, n_topics). Useful both for visual inspection in the
        notebook and for export to CSV when annotating topics by hand.
    """
    top = model.get_top_words(n_words=n_words)
    columns = _resolve_labels(topic_labels, model.n_topics, prefix="Topic ")
    return pd.DataFrame(np.array(top).T, columns=columns)


def make_label_template(model, n_words=10, save_path=None):
    """Print a fill-in-the-blank template for manual topic labelling.

    The output looks like:

        Topic  0 [maire, majorité, gauche, liberté, ...] -> __________
        Topic  1 [écologie, société, écologiste, ...]    -> __________
        ...

    which is the form I find most useful when sitting down to annotate
    topics: the top words are right next to the blank that has to be filled.

    If `save_path` is given, the template is also written to a text file
    (handy to keep alongside the model artifacts).
    """
    top = model.get_top_words(n_words=n_words)
    lines = []
    for k, words in enumerate(top):
        words_str = ", ".join(words)
        lines.append(f"Topic {k:>2} [{words_str}] -> __________")
    template = "\n".join(lines)
    print(template)
    if save_path is not None:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(template, encoding="utf-8")
    return template


def apply_labels(labels_dict, n_topics, fallback="Topic {k}"):
    """Turn a {topic_id: name} dict into a list of length `n_topics`.

    Topics not present in `labels_dict` get a placeholder produced from
    `fallback` (a `.format`-style template with `{k}` for the topic index).
    """
    out = []
    for k in range(n_topics):
        if k in labels_dict:
            out.append(labels_dict[k])
        else:
            out.append(fallback.format(k=k))
    return out


# =============================================================================
# Representative documents
# =============================================================================

def find_representative_documents(
    model,
    df,
    text_col="text",
    n_docs=3,
    n_chars=400,
    start_char=0,
    topic_labels=None,
    extra_cols=("year", "titulaire-soutien"),
    return_df=True,
):
    """For each topic, return the `n_docs` documents with the highest weight.

    Inspecting these documents alongside the top words is the standard way
    to validate a topic name: if the proposed label fits both the top words
    AND the manifestos that load most strongly on the topic, it is safe to
    keep; otherwise the topic is probably mixed and the label has to change.

    Parameters
    ----------
    model : TopicModelBase
        Fitted topic model.
    df : pd.DataFrame
        Original DataFrame, aligned row-by-row with the documents fed to the
        model (i.e. df_clean in the notebook).
    text_col : str
        Column with the raw text to display.
    n_docs : int
        Number of representative documents per topic.
    n_chars : int
        Number of characters of the raw text to print as preview.
    topic_labels : list[str] or dict[int, str], optional
        Display labels for the topics. If None, defaults to "Topic k".
    extra_cols : tuple of str
        Metadata columns to print alongside the preview (skipped silently
        if missing from `df`).
    return_df : bool
        If True (default), also return a DataFrame with topic_id, rank,
        doc_index, weight and the metadata columns. The print is always done.
    """
    doc_topic = model.get_doc_topic_matrix()
    if doc_topic.shape[0] != len(df):
        raise ValueError(
            f"doc_topic has {doc_topic.shape[0]} rows but df has {len(df)} — "
            "they must be aligned. Did you slice df after fitting the model?"
        )

    n_topics = model.n_topics
    labels = _resolve_labels(topic_labels, n_topics, prefix="Topic ")
    rows = []
    for k in range(n_topics):
        # argsort descending, take top n_docs
        top_idx = np.argsort(doc_topic[:, k])[::-1][:n_docs]
        print("=" * 88)
        print(f"{labels[k]}  (top {n_docs} representative documents)")
        print("=" * 88)
        for rank, idx in enumerate(top_idx, start=1):
            weight = float(doc_topic[idx, k])
            print(f"\n  [{rank}] doc index = {idx}, topic weight = {weight:.3f}")
            for col in extra_cols:
                if col in df.columns:
                    print(f"      {col:18s}: {df.iloc[idx][col]}")
            preview = str(df.iloc[idx][text_col])[start_char:start_char + n_chars].replace("\n", " ")
            print(f"      preview: {preview}...")
            row = {
                "topic_id": k,
                "topic_label": labels[k],
                "rank": rank,
                "doc_index": int(idx),
                "weight": weight,
            }
            for col in extra_cols:
                if col in df.columns:
                    row[col] = df.iloc[idx][col]
            rows.append(row)
        print()
    if return_df:
        return pd.DataFrame(rows)
    return None


# =============================================================================
# Cross-model alignment (LDA <-> NMF)
# =============================================================================

def align_topics(top_words_a, top_words_b, n_words=15, method="jaccard"):
    """Match each topic of model A to its closest topic in model B.

    Useful to answer "is this LDA topic also recovered by NMF?". We compare
    topics by their top-N word sets only (the doc-topic matrices live in
    different spaces — a probability simplex for LDA, a normalised NMF
    activation for NMF — so word-set overlap is the most direct comparison).

    Parameters
    ----------
    top_words_a, top_words_b : list of list of str
        Top words per topic, as returned by `model.get_top_words()`.
    n_words : int
        Truncate both lists to the top `n_words`.
    method : {"jaccard", "overlap"}
        - "jaccard" : |A ∩ B| / |A ∪ B|  (penalises asymmetric topics)
        - "overlap" : |A ∩ B| / min(|A|, |B|)  (more lenient)

    Returns
    -------
    similarity : np.ndarray of shape (n_topics_a, n_topics_b)
        Pairwise similarities.
    matches : pd.DataFrame
        For each topic of A, its best match in B with the similarity score
        and the shared words.
    """
    sets_a = [set(t[:n_words]) for t in top_words_a]
    sets_b = [set(t[:n_words]) for t in top_words_b]
    na, nb = len(sets_a), len(sets_b)
    sim = np.zeros((na, nb), dtype=float)
    for i, sa in enumerate(sets_a):
        for j, sb in enumerate(sets_b):
            inter = sa & sb
            if not inter:
                continue
            if method == "jaccard":
                sim[i, j] = len(inter) / len(sa | sb)
            elif method == "overlap":
                sim[i, j] = len(inter) / min(len(sa), len(sb))
            else:
                raise ValueError(f"Unknown method '{method}'")

    rows = []
    for i in range(na):
        j = int(sim[i].argmax())
        shared = sorted(sets_a[i] & sets_b[j])
        rows.append({
            "topic_a":      i,
            "best_match_b": j,
            "similarity":   float(sim[i, j]),
            "n_shared":     len(shared),
            "shared_words": ", ".join(shared),
        })
    return sim, pd.DataFrame(rows)


# =============================================================================
# Internal helpers
# =============================================================================

def _resolve_labels(topic_labels, n_topics, prefix="Topic "):
    """Normalise the various forms accepted for `topic_labels`."""
    if topic_labels is None:
        return [f"{prefix}{k}" for k in range(n_topics)]
    if isinstance(topic_labels, dict):
        return apply_labels(topic_labels, n_topics, fallback=prefix + "{k}")
    if len(topic_labels) != n_topics:
        raise ValueError(
            f"topic_labels has length {len(topic_labels)} but model has "
            f"{n_topics} topics."
        )
    return list(topic_labels)
