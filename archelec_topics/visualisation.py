"""Visualization functions for topic models.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

import numpy as np
import matplotlib.pyplot as plt
from archelec_topics.interpretation import _resolve_labels
from wordcloud import WordCloud

def plot_topic_words(
    source,
    title="",
    n_cols=5,
    n_words=12,
    topic_labels=None,
    xlabel=None,
    color="#332288",
):
    """Bar plot of top words per topic, with bar length = real word weight.

    The previous version used `range(n, 0, -1)` as bar lengths, i.e. a purely
    rank-based pseudo-weight. This version uses the model's actual weights so
    the x-axis is interpretable:
      - LDA  -> P(word | topic)   (gensim's `show_topic` probabilities)
      - NMF  -> entries of the H components matrix
      - BERTopic -> c-TF-IDF scores

    Parameters
    ----------
    source :
        - a fitted topic model exposing `_topic_word_weights`-compatible
          attributes (LDATopicModel, NMFTopicModel, BERTopicModel), OR
        - a list of {word: weight} dicts (one per topic), OR
        - a list of word lists (legacy input — falls back to rank pseudo-weights
          and warns via the x-axis label).
    n_words : int
        Number of top words to display per topic.
    xlabel : str, optional
        Override the auto-detected x-axis label.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # --- 1. Normalize input to a list of {word: weight} dicts -----------------
    if hasattr(source, "n_topics"):
        # A fitted model -> use the existing helper to get real weights
        weights_per_topic = _topic_word_weights(source, n_words=n_words)
        if xlabel is None:
            if hasattr(source, "model") and hasattr(source.model, "show_topic"):
                xlabel = r"$P(\mathrm{word}\,|\,\mathrm{topic})$"
            elif hasattr(source, "_H"):
                xlabel = "NMF component weight"
            else:
                xlabel = "Weight"
    elif len(source) > 0 and isinstance(source[0], dict):
        # Already weight dicts
        weights_per_topic = [
            dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n_words])
            for d in source
        ]
        if xlabel is None:
            xlabel = "Weight"
    else:
        # Legacy: list of word lists, no weights available
        weights_per_topic = [
            {w: float(len(words) - i) for i, w in enumerate(words[:n_words])}
            for words in source
        ]
        if xlabel is None:
            xlabel = "Rank (no weights provided)"

    # --- 2. Layout ------------------------------------------------------------
    n_topics = len(weights_per_topic)
    labels = _resolve_labels(topic_labels, n_topics, prefix="Topic ")
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # --- 3. One bar plot per topic, sorted by descending weight ---------------
    for k, weights in enumerate(weights_per_topic):
        ax = axes[k]
        items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        words = [w for w, _ in items]
        vals = [v for _, v in items]
        n = len(items)

        ax.barh(range(n), vals, color=color)
        ax.set_yticks(range(n))
        ax.set_yticklabels(words, fontsize=8)
        ax.invert_yaxis()                     # highest weight on top
        ax.set_title(labels[k], fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)

    # Hide unused cells
    for k in range(n_topics, len(axes)):
        axes[k].axis("off")

    # x-label only on the bottom row to keep the figure clean
    bottom_row_start = (n_rows - 1) * n_cols
    for k in range(bottom_row_start, n_topics):
        axes[k].set_xlabel(xlabel, fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


def plot_topic_wordclouds(
    model,
    n_words=30,
    n_cols=5,
    title="",
    topic_labels=None,
    colormap="viridis",
    background_color="white",
    figsize_per_cell=(3.2, 2.4),
):
    """Word cloud per topic, with word size proportional to model weight.

    LDA topics expose word probabilities directly via `show_topic`; NMF
    topics expose word weights via the components matrix `_H`. Both yield a
    {word: weight} dict that the wordcloud library can consume.

    Requires the `wordcloud` package (`pip install wordcloud`). If it is not
    installed, an informative error is raised.
    """

    weights_per_topic = _topic_word_weights(model, n_words=n_words)
    n_topics = len(weights_per_topic)
    labels = _resolve_labels(topic_labels, n_topics, prefix="Topic ")
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
    )
    axes = np.atleast_1d(axes).flatten()
    for k, weights in enumerate(weights_per_topic):
        wc = WordCloud(
            width=400, height=300,
            background_color=background_color,
            colormap=colormap,
            prefer_horizontal=0.95,
            relative_scaling=0.5,
            random_state=42,
        ).generate_from_frequencies(weights)
        axes[k].imshow(wc, interpolation="bilinear")
        axes[k].set_title(labels[k], fontsize=10)
        axes[k].axis("off")
    for k in range(n_topics, len(axes)):
        axes[k].axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig


# =============================================================================
# Doc-topic heatmap (Blei 2012-style, à la slides)
# =============================================================================

def plot_doc_topic_heatmap(
    doc_topic_matrix,
    sample_size=200,
    sort_by="dominant_topic",
    topic_labels=None,
    title="Document × topic distribution",
    cmap="magma",
    random_state=42,
):
    """Heatmap of a document-by-topic matrix to visualize sparsity.

    Following the Blei (2012) figure reproduced in the course slides ("Documents
    exhibit multiple topics"), this plot answers two questions at once:
    - are topics well-separated, or do most documents have a flat distribution
      over many topics? (a flat distribution = LDA without enough Dirichlet
      sparsity, or NMF with poor reconstruction)
    - which topics dominate the corpus and which are rare?

    Parameters
    ----------
    doc_topic_matrix : np.ndarray
        Shape (n_docs, n_topics). Output of `model.get_doc_topic_matrix()`.
    sample_size : int
        Number of documents to display (the matrix is too large to plot
        entirely). Drawn at random if `sort_by` is None, otherwise sorted.
    sort_by : {"dominant_topic", None}
        If "dominant_topic", documents are sorted by their argmax topic so
        that the heatmap shows visible blocks of dominant topics — useful to
        spot redundant or empty topics.
    """
    rng = np.random.default_rng(random_state)
    n_docs, n_topics = doc_topic_matrix.shape
    if sample_size and sample_size < n_docs:
        idx = rng.choice(n_docs, size=sample_size, replace=False)
    else:
        idx = np.arange(n_docs)
    M = doc_topic_matrix[idx]

    if sort_by == "dominant_topic":
        dominant = M.argmax(axis=1)
        order = np.argsort(dominant)
        M = M[order]

    labels = _resolve_labels(topic_labels, n_topics, prefix="T")
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * n_topics + 4), 5))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=M.max())
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Topic")
    ax.set_ylabel(f"Document (sample of {len(M)})")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Topic weight")
    plt.tight_layout()
    return fig


# =============================================================================
# Topic prevalence over time
# =============================================================================

def plot_topic_prevalence_over_time(
    doc_topic_matrix,
    year_series,
    topic_labels=None,
    title="Topic prevalence across legislative elections",
    kind="line",
    figsize=(8, 4.5),
):
    """Mean topic weight per year — natural transition to the DTM analysis.

    For each year, takes the average topic distribution across all documents
    of that year. Two display modes:
      - "line" : one line per topic, useful when there are few topics
      - "stack": stacked area, useful to see relative composition

    Parameters
    ----------
    doc_topic_matrix : np.ndarray  (n_docs, n_topics)
    year_series : pd.Series  (length n_docs)
        Year of each document, aligned with `doc_topic_matrix`.
    """
    import pandas as pd
    n_docs, n_topics = doc_topic_matrix.shape
    if len(year_series) != n_docs:
        raise ValueError("year_series and doc_topic_matrix must have the same length")

    labels = _resolve_labels(topic_labels, n_topics, prefix="Topic ")
    df = pd.DataFrame(doc_topic_matrix, columns=labels)
    df["year"] = pd.Series(year_series).values
    prevalence = df.dropna(subset=["year"]).groupby("year").mean()
    prevalence.index = prevalence.index.astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    if kind == "stack":
        prevalence.plot(kind="area", stacked=True, ax=ax, alpha=0.8,
                        colormap="tab10")
    else:
        prevalence.plot(ax=ax, marker="o", linewidth=1.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean topic weight")
    ax.set_title(title)
    ax.set_xticks(prevalence.index)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              title="Topic")
    plt.tight_layout()
    return fig, prevalence


# =============================================================================
# LDA / NMF alignment heatmap
# =============================================================================

def plot_topic_alignment_matrix(
    top_words_a,
    top_words_b,
    name_a="LDA",
    name_b="NMF",
    n_words=15,
    method="jaccard",
    labels_a=None,
    labels_b=None,
    title=None,
):
    """Heatmap of pairwise topic similarities between two models."""
    from archelec_topics.interpretation import align_topics
    sim, _ = align_topics(top_words_a, top_words_b, n_words=n_words, method=method)
    na, nb = sim.shape
    la = _resolve_labels(labels_a, na, prefix=f"{name_a} ")
    lb = _resolve_labels(labels_b, nb, prefix=f"{name_b} ")

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * nb + 2), max(4, 0.6 * na + 2)))
    im = ax.imshow(sim, aspect="auto", cmap="Blues", vmin=0, vmax=sim.max())
    ax.set_xticks(range(nb))
    ax.set_xticklabels(lb, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(na))
    ax.set_yticklabels(la, fontsize=8)
    ax.set_xlabel(name_b)
    ax.set_ylabel(name_a)
    if title is None:
        title = f"Topic similarity ({method}, top {n_words} words)"
    ax.set_title(title)

    # annotate non-trivial cells
    thr = max(0.05, sim.max() * 0.25)
    for i in range(na):
        for j in range(nb):
            if sim[i, j] >= thr:
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                        color="white" if sim[i, j] > sim.max() * 0.5 else "black",
                        fontsize=7)
    fig.colorbar(im, ax=ax, label=f"{method} similarity")
    plt.tight_layout()
    return fig

# =============================================================================
# BERTopic — static plot of `topics_over_time`
# =============================================================================

def plot_bertopic_topics_over_time(
    topics_over_time_df,
    top_n_topics=8,
    topic_labels=None,
    title="Topic prevalence across legislative elections (BERTopic)",
    figsize=(8, 4.5),
    normalize=True,
):
    """Static line plot of BERTopic's `topics_over_time` DataFrame.

    BERTopic's native `visualize_topics_over_time` uses Plotly, which doesn't
    render in a static PDF report; this helper produces the matplotlib
    equivalent in the project's NeurIPS style.

    Parameters
    ----------
    topics_over_time_df : pd.DataFrame
        Output of `BERTopicModel.topics_over_time(...)`. Columns:
        Topic, Words, Frequency, Timestamp.
    top_n_topics : int
        Display only the `top_n_topics` largest topics overall (in document
        count) — plotting all 10+ topics on the same axes is unreadable.
    topic_labels : dict[int, str] or list[str], optional
        Manual labels keyed by raw topic id (dict) or by ordinal position
        in the kept-topic list (list).
    normalize : bool
        If True, plot the share of each topic per timestep (each timestep
        sums to 1). If False, plot the absolute document count.
    """
    df = topics_over_time_df.copy()
    totals = df.groupby("Topic")["Frequency"].sum().sort_values(ascending=False)
    keep = totals.head(top_n_topics).index.tolist()
    df = df[df["Topic"].isin(keep)]

    pivot = df.pivot_table(
        index="Timestamp", columns="Topic", values="Frequency", fill_value=0,
    )
    if normalize:
        pivot = pivot.div(pivot.sum(axis=1), axis=0)

    if isinstance(topic_labels, dict):
        col_labels = {tid: topic_labels.get(tid, f"Topic {tid}") for tid in pivot.columns}
    elif topic_labels is not None:
        col_labels = {tid: topic_labels[i] for i, tid in enumerate(pivot.columns)}
    else:
        col_labels = {tid: f"Topic {tid}" for tid in pivot.columns}
    pivot = pivot.rename(columns=col_labels)

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(ax=ax, marker="o", linewidth=1.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of documents" if normalize else "Document count")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
              title="Topic")
    plt.tight_layout()
    return fig, pivot


# =============================================================================
# Internal helpers
# =============================================================================

def _topic_word_weights(model, n_words=30):
    """Extract {word: weight} dicts per topic from either an LDA or NMF model.

    Tries the model-specific accessors first, falls back to a uniform 1/rank
    weighting if the underlying model doesn't expose probabilities (defensive,
    so the function still works if the model interface evolves).
    """
    out = []
    n_topics = model.n_topics

    # LDA: gensim's show_topic returns (word, prob) pairs directly
    if hasattr(model, "model") and hasattr(model.model, "show_topic"):
        for k in range(n_topics):
            pairs = model.model.show_topic(k, topn=n_words)
            out.append({w: float(p) for w, p in pairs})
        return out

    # NMF: read components_ via the wrapper attributes
    if hasattr(model, "_H") and hasattr(model, "feature_names_"):
        H = model._H
        names = model.feature_names_
        for k in range(n_topics):
            top_idx = H[k].argsort()[::-1][:n_words]
            out.append({names[i]: float(H[k, i]) for i in top_idx})
        return out

    # Fallback: rank-based pseudo-weights from get_top_words
    for words in model.get_top_words(n_words=n_words):
        out.append({w: float(n_words - i) for i, w in enumerate(words)})
    return out