"""Topic-by-metadata analyses: how do topics distribute over parties / families?

This module sits one layer above topic models: it takes a fitted model's
doc-topic matrix and a categorical grouping (parties or political families,
encoded as 0/1 dummy columns by `parties_processing.add_party_columns`) and
computes:

  - the **mean topic profile** of each group (DataFrame group × topic),
  - the **top topics per group** (which topics each group endorses most),
  - **heatmap** visualisations of the above (raw or row-normalised),
  - **bar charts** of the topic profile of a single party / family,

with a unified API (`groups=`/`top_n=`) that works for both parties and
political families. The two convenience wrappers `topics_by_party` and
`topics_by_family` are thin shims around the generic functions and the only
difference is the dummies they consume.

A document that supports a coalition (e.g. PCF + PS in 1973) contributes to
**both** parties' profiles, by design — that's the same convention used in
the stacked-area plots of the notebook (cells 22 and 23) and matches how
`add_party_columns` builds the dummies.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from archelec_topics.interpretation import _resolve_labels


# =============================================================================
# Core: mean topic distribution per group
# =============================================================================

def topic_distribution_by_group(
    doc_topic_matrix,
    group_dummies,
    groups=None,
    min_docs=10,
    topic_labels=None,
):
    """Mean topic weight per group.

    For each group g, this computes the average topic distribution restricted
    to documents that endorse g (i.e. with `group_dummies[g] == 1`). Groups
    with fewer than `min_docs` supporting manifestos are dropped, since their
    mean profile would be too noisy to interpret.

    Parameters
    ----------
    doc_topic_matrix : np.ndarray  shape (n_docs, n_topics)
        Output of `model.get_doc_topic_matrix()`.
    group_dummies : pd.DataFrame  shape (n_docs, n_groups)
        0/1 dummies, one column per group. Typically `party_dummies` or
        `family_dummies` from `add_party_columns`.
    groups : list[str] or None
        Subset / order of groups to keep. If None, all columns are used.
    min_docs : int
        Minimum number of supporting documents for a group to be included.
    topic_labels : list[str] / dict[int, str] / None
        Display labels for topics, used as columns of the output.

    Returns
    -------
    pd.DataFrame
        Index: group name. Columns: topic label. Values: mean topic weight.
        Each row is a probability-like vector that sums approximately to 1
        (exactly 1 for LDA, also 1 for NMF since `get_doc_topic_matrix` is
        normalised in the wrapper).
    n_docs_per_group : pd.Series
        Number of documents supporting each kept group.
    """
    if doc_topic_matrix.shape[0] != len(group_dummies):
        raise ValueError(
            f"doc_topic_matrix has {doc_topic_matrix.shape[0]} rows but "
            f"group_dummies has {len(group_dummies)} — they must align."
        )

    if groups is None:
        groups = list(group_dummies.columns)

    n_topics = doc_topic_matrix.shape[1]
    labels = _resolve_labels(topic_labels, n_topics, prefix="Topic ")

    rows = {}
    counts = {}
    for g in groups:
        if g not in group_dummies.columns:
            print(f"[warn] group '{g}' not in group_dummies — skipped")
            continue
        mask = group_dummies[g].values.astype(bool)
        n = int(mask.sum())
        if n < min_docs:
            print(f"[skip] '{g}' has only {n} documents (< min_docs={min_docs})")
            continue
        rows[g] = doc_topic_matrix[mask].mean(axis=0)
        counts[g] = n

    if not rows:
        raise ValueError("No group passed the min_docs threshold.")
    out = pd.DataFrame(rows, index=labels).T
    out.index.name = "group"
    return out, pd.Series(counts, name="n_docs")


def top_topics_per_group(group_topic_dist, top_n=3):
    """For each group, return the `top_n` topics with the highest mean weight.

    Returns a long-form DataFrame (group, rank, topic, weight) ready to print
    in the notebook or to export. The topic names come directly from the
    columns of `group_topic_dist`, so they will be the human labels if the
    upstream call passed `topic_labels=`.
    """
    rows = []
    for g, profile in group_topic_dist.iterrows():
        top = profile.sort_values(ascending=False).head(top_n)
        for rank, (topic, w) in enumerate(top.items(), start=1):
            rows.append({"group": g, "rank": rank, "topic": topic, "weight": float(w)})
    return pd.DataFrame(rows)


# =============================================================================
# Visualisations
# =============================================================================

def plot_topic_group_heatmap(
    group_topic_dist,
    title="Topic prevalence by group",
    normalize=None,
    annotate=True,
    cmap="magma",
    figsize=None,
):
    """Heatmap of `group_topic_dist` (groups × topics).

    Parameters
    ----------
    normalize : {"row", "col", "z", None}
        - None  : raw mean weights (default, comparable across groups)
        - "row" : each row sums to 1 (= relative composition within a group;
                  by construction this is roughly already the case, but
                  re-normalising removes any residual <1 from minimum_probability
                  thresholds)
        - "col" : each column sums to 1 (= which group "owns" each topic;
                  emphasizes specialisation)
        - "z"   : per-column z-score (= which groups over- or under-use each
                  topic relative to the average — most useful for spotting
                  party-specific topics)
    annotate : bool
        Print the value in each cell.
    """
    M = group_topic_dist.copy()
    if normalize == "row":
        M = M.div(M.sum(axis=1), axis=0)
    elif normalize == "col":
        col_sum = M.sum(axis=0).replace(0, 1)
        M = M.div(col_sum, axis=1)
    elif normalize == "z":
        mu = M.mean(axis=0); sd = M.std(axis=0).replace(0, 1)
        M = (M - mu) / sd
        cmap = "RdBu_r"   # diverging palette is more readable for z-scores

    n_rows, n_cols = M.shape
    if figsize is None:
        figsize = (max(6, 0.6 * n_cols + 3), max(3, 0.45 * n_rows + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    if normalize == "z":
        vmax = max(abs(M.values.min()), abs(M.values.max()))
        im = ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(M.values, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(M.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(M.index, fontsize=9)
    ax.set_title(title)
    if annotate:
        thr = (M.values.max() + M.values.min()) / 2
        for i in range(n_rows):
            for j in range(n_cols):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > thr else "black")
    label = {"row": "share within group", "col": "share within topic",
             "z": "z-score", None: "mean topic weight"}.get(normalize, "value")
    fig.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    return fig


def plot_group_topic_profile(
    group_topic_dist,
    group_name,
    title=None,
    color="#332288",
    figsize=(7, 3.2),
):
    """Bar chart of one group's mean topic profile.

    Companion to the heatmap when one wants to focus on a single party or
    family in detail (e.g. for a specific paragraph in the report).
    """
    if group_name not in group_topic_dist.index:
        raise KeyError(
            f"'{group_name}' not in group_topic_dist. Available: "
            f"{list(group_topic_dist.index)[:10]}..."
        )
    profile = group_topic_dist.loc[group_name].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(profile)), profile.values, color=color)
    ax.set_xticks(range(len(profile)))
    ax.set_xticklabels(profile.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean topic weight")
    ax.set_title(title or f"Topic profile — {group_name}")
    plt.tight_layout()
    return fig


# =============================================================================
# Convenience wrappers — parties and families
# =============================================================================

def topics_by_party(
    doc_topic_matrix,
    party_dummies,
    parties=None,
    top_n_parties=10,
    party_counts=None,
    topic_labels=None,
    min_docs=10,
):
    """Topic distribution for a list of parties.

    Two modes:
    - if `parties` is given, it's the list to analyse;
    - otherwise the top `top_n_parties` parties (by number of supporting
      documents) are selected automatically. The selection uses
      `party_counts` if provided (the `diagnostic["party_counts"]` Counter
      from `add_party_columns` works directly), and falls back to summing
      the dummies.

    Returns the same `(group_topic_dist, n_docs_per_group)` pair as the
    underlying generic function.
    """
    if parties is None:
        if party_counts is not None:
            ranked = [p for p, _ in party_counts.most_common()
                      if p in party_dummies.columns]
        else:
            ranked = party_dummies.sum(axis=0).sort_values(ascending=False).index.tolist()
        parties = ranked[:top_n_parties]

    return topic_distribution_by_group(
        doc_topic_matrix, party_dummies,
        groups=parties, min_docs=min_docs, topic_labels=topic_labels,
    )


def topics_by_family(
    doc_topic_matrix,
    family_dummies,
    families=None,
    topic_labels=None,
    min_docs=10,
):
    """Topic distribution for political families (8 + 'Divers').

    By default takes all columns of `family_dummies` (which is already in the
    canonical FAMILY_ORDER from `parties_processing`). Pass `families=` to
    restrict / reorder.
    """
    return topic_distribution_by_group(
        doc_topic_matrix, family_dummies,
        groups=families, min_docs=min_docs, topic_labels=topic_labels,
    )
