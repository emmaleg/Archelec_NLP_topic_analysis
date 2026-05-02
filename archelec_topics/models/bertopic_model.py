"""BERTopic wrapper following the same TopicModelBase interface as LDA/NMF.

BERTopic (Grootendorst, 2022) is the natural successor to the LDA/NMF analyses
done in §2: it replaces the bag-of-words assumption by Sentence-BERT embeddings,
clusters the resulting embeddings with HDBSCAN, and re-extracts topic words via
a class-based TF-IDF (c-TF-IDF) on the clusters. It is the modern endpoint of
the topic-modelling timeline shown in the course (slide "Topic: History (until
BERT)") and produces sharper, semantically-aware topics than LDA/NMF on
medium-length texts.

Two design notes
----------------

1. **Input split** — BERTopic best practice is to feed *raw* text to the
   embedding model (BERT understands capitalisation, context, n-grams natively)
   while the c-TF-IDF step uses a CountVectorizer with custom stopwords. The
   wrapper accepts raw strings and applies a CountVectorizer built from the
   project's `EXTRA_STOPWORDS` for the topic-words extraction.

2. **Outlier topic (-1)** — HDBSCAN labels documents not assigned to any
   cluster as topic -1. To stay compatible with the rest of the pipeline
   (`topic_metadata_analysis`, `find_representative_documents`...), the
   wrapper exposes only the non-outlier topics and represents the doc-topic
   matrix as a hard one-hot encoding of cluster assignments (rows of all
   zeros for outlier docs). Mean topic weight per group then reads naturally
   as "fraction of group documents dominated by each topic".

Heavy dependencies (`bertopic`, `sentence-transformers`, `umap-learn`,
`hdbscan`, `torch`) are imported inside `fit()` and inside the
embedding helper.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from archelec_topics.models.base import TopicModelBase


# =============================================================================
# Embedding helper — kept separate so it can be pickled/cached independently
# =============================================================================

def compute_embeddings(
    texts,
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    cache_path=None,
    batch_size=32,
    show_progress_bar=True,
):
    """Compute Sentence-BERT embeddings for a list of texts, with caching.

    The embedding step is by far the slowest part of the BERTopic pipeline
    on a CPU. Cached as a pickle next to the tokens so it survives kernel
    restarts.

    On Apple Silicon the `sentence-transformers` library auto-detects MPS
    and runs ~5x faster than CPU; on a CUDA GPU it auto-detects the device.

    Parameters
    ----------
    texts : list[str]
        Raw documents (NOT lemmatised — BERT works on raw text).
    model_name : str
        A SentenceTransformer-compatible model. The default is the
        multilingual MiniLM, which handles French well and is small (~120MB).
        For higher quality at the cost of slower embedding, switch to
        "paraphrase-multilingual-mpnet-base-v2" (~970MB).
    cache_path : str | Path | None
        If given, embeddings are saved here and reloaded on subsequent calls.
    """
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        if len(embeddings) == len(texts):
            print(f"Loaded {len(embeddings)} cached embeddings from {cache_path}")
            return embeddings
        print(f"[warn] cache size mismatch ({len(embeddings)} vs {len(texts)}), recomputing")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "compute_embeddings requires `sentence-transformers`. "
            "Install with `pip install sentence-transformers`."
        ) from e

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
    )

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved {len(embeddings)} embeddings to {cache_path}")

    return embeddings


# =============================================================================
# The wrapper
# =============================================================================

class BERTopicModel(TopicModelBase):
    """Wrapper around `bertopic.BERTopic`.

    Parameters
    ----------
    n_topics : int or "auto"
        - int: target number of topics (BERTopic merges similar clusters
          until it reaches this count). Match the K used for LDA/NMF for
          fair comparison (e.g. 10).
        - "auto": let HDBSCAN decide; typically yields 30–80 topics on a
          corpus of this size, useful for fine-grained exploration.
    embedding_model : str
        Sentence-BERT model. Same default as `compute_embeddings`.
    min_topic_size : int
        Minimum cluster size for HDBSCAN. Larger = fewer, broader topics.
        15 is BERTopic's default; 30 is a sensible value for ~21k docs.
    extra_stopwords : iterable[str] or None
        Stopwords used inside the c-TF-IDF CountVectorizer (only affects
        the topic-words extraction step, not the embedding). Pass
        `EXTRA_STOPWORDS` from `preprocessing` to stay consistent with the
        LDA/NMF runs.
    n_gram_range : (int, int)
        Word n-gram range for the c-TF-IDF step. (1, 2) extracts unigrams
        and bigrams from the topic words, useful to recover compound
        political terms that didn't make it into the manual `_COMPOUND_PATTERNS`.
    random_state : int
        Seed forwarded to UMAP (HDBSCAN is deterministic).
    """

    def __init__(
        self,
        n_topics=10,
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        min_topic_size=30,
        extra_stopwords=None,
        min_df = 15,
        max_df = 0.6,
        max_features = 10000,
        n_gram_range=(1, 2),
        random_state=42,
        umap_n_neighbors=15,
        umap_n_components=5,
        verbose=True,
    ):
        # n_topics in the base class is the target; "auto" stored as None
        target = None if n_topics == "auto" else n_topics
        super().__init__(n_topics=target if target is not None else 0,
                         random_state=random_state)
        self._target_n_topics  = target
        self.embedding_model   = embedding_model
        self.min_topic_size    = min_topic_size
        self.min_df            = min_df
        self.max_df            = max_df
        self.max_features      = self.max_features
        self.extra_stopwords   = list(extra_stopwords) if extra_stopwords is not None else None
        self.n_gram_range      = n_gram_range
        self.umap_n_neighbors  = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.verbose           = verbose

        self.model           = None
        self._documents      = None
        self._topic_assignments = None  # raw output incl. -1
        self._kept_topic_ids = None     # ordered ids excluding -1

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, documents, embeddings=None, lemmatized_documents=None, **kwargs):
        """Fit BERTopic.

        Parameters
        ----------
        documents : list[str]
            Raw documents. Pass `df_clean["text"].tolist()`, NOT the
            lemmatised tokens used for LDA/NMF.
        embeddings : np.ndarray, optional
            Pre-computed embeddings, aligned with `documents`. Strongly
            recommended for iterative work — see `compute_embeddings`.
        """
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
            from hdbscan import HDBSCAN
        except ImportError as e:
            raise ImportError(
                "BERTopicModel requires `bertopic`, `sentence-transformers`, "
                "`umap-learn` and `hdbscan`. Install with:\n"
                "    pip install bertopic sentence-transformers"
            ) from e

        if not isinstance(documents[0], str):
            raise ValueError(
                "BERTopicModel expects raw text (list[str]), not tokens. "
                "Pass df_clean['text'].tolist(), not the preprocessed tokens."
            )
        self._documents = list(documents)

        # Custom UMAP with reproducibility
        umap_model = UMAP(
            n_neighbors  = self.umap_n_neighbors,
            n_components = self.umap_n_components,
            min_dist     = 0.0,
            metric       = "cosine",
            random_state = self.random_state,
        )

        # HDBSCAN — controls how many tiny clusters get merged into outliers
        hdbscan_model = HDBSCAN(
            min_cluster_size       = self.min_topic_size,
            metric                 = "euclidean",
            cluster_selection_method = "eom",
            prediction_data        = True,
        )

        # CountVectorizer for c-TF-IDF — same stopwords as LDA/NMF
        if self.extra_stopwords is not None:
            import spacy
            from archelec_topics.preprocessing import strip_accents
            nlp = spacy.load("fr_core_news_md", disable=["ner", "parser", "tagger"])
            full_sw = set(nlp.Defaults.stop_words) | set(self.extra_stopwords)
            stopwords_list = sorted({strip_accents(w.lower()) for w in full_sw})
        else:
            stopwords_list = None
        
        vectorizer_model = CountVectorizer(
            ngram_range  = self.n_gram_range,
            stop_words   = stopwords_list,
            min_df       = self.min_df,
            max_df       = self.max_df,
            max_features = self.max_features, 
        )

        self.model = BERTopic(
            embedding_model   = self.embedding_model,
            umap_model        = umap_model,
            hdbscan_model     = hdbscan_model,
            vectorizer_model  = vectorizer_model,
            nr_topics         = self._target_n_topics,   # None or int
            calculate_probabilities = False,             # too slow; we use hard labels
            verbose           = self.verbose,
        )

        topics, _ = self.model.fit_transform(self._documents, embeddings=embeddings)
        self._topic_assignments = np.asarray(topics)

        # Best practice non-anglophone (Grootendorst) : embeddings sur texte brut,
        # c-TF-IDF sur texte lemmatisé. update_topics ne refait pas le clustering,
        # il ne fait que ré-extraire les top words depuis les nouveaux documents.
        if lemmatized_documents is not None:
            if len(lemmatized_documents) != len(self._documents):
                raise ValueError(
                    f"lemmatized_documents has length {len(lemmatized_documents)} "
                    f"but documents has {len(self._documents)}."
                )
            self.model.update_topics(
                lemmatized_documents,
                vectorizer_model=vectorizer_model,
            )

        # Determine the kept topic IDs (exclude -1)
        info = self.model.get_topic_info()
        self._kept_topic_ids = [t for t in info["Topic"].tolist() if t != -1]
        self.n_topics = len(self._kept_topic_ids)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    def get_doc_topic_matrix(self):
        """One-hot doc-topic matrix (n_docs × n_topics), outliers = all zeros.

        For BERTopic the natural representation is a hard cluster assignment.
        Soft probabilities can be obtained via `model.approximate_distribution`
        but it's slow and not needed for the downstream analyses.
        """
        self._check_fitted()
        n_docs = len(self._topic_assignments)
        out = np.zeros((n_docs, self.n_topics), dtype=float)
        # Map raw topic id -> column index
        id_to_col = {tid: i for i, tid in enumerate(self._kept_topic_ids)}
        for i, t in enumerate(self._topic_assignments):
            j = id_to_col.get(int(t))
            if j is not None:
                out[i, j] = 1.0
        return out

    def get_top_words(self, n_words=10):
        """Top words per topic via c-TF-IDF, in the same order as `_kept_topic_ids`."""
        self._check_fitted()
        out = []
        for tid in self._kept_topic_ids:
            words = self.model.get_topic(tid)
            if not words:
                out.append([])
                continue
            out.append([w for w, _ in words[:n_words]])
        return out

    # ------------------------------------------------------------------
    # BERTopic-specific extras
    # ------------------------------------------------------------------

    def get_topic_info(self):
        """Pass-through to BERTopic's topic info table, restricted to kept topics."""
        self._check_fitted()
        info = self.model.get_topic_info()
        return info[info["Topic"] != -1].reset_index(drop=True)

    def n_outliers(self):
        """Number of documents assigned to topic -1 (outliers)."""
        self._check_fitted()
        return int((self._topic_assignments == -1).sum())

    def topics_over_time(self, timestamps, nr_bins=None, evolution_tuning=True):
        """Compute topic prevalence over time using BERTopic's native method.

        Parameters
        ----------
        timestamps : list / pd.Series of length n_docs
            Year of each document, aligned with the documents passed to fit().
        nr_bins : int, optional
            Number of time bins. If None, uses unique timestamp values
            (so 5 bins for the 5 election years).
        evolution_tuning : bool
            BERTopic option that smooths topic word weights over time using
            cosine similarity between consecutive bins. Useful to make the
            evolution chart less noisy.

        Returns
        -------
        pd.DataFrame
            Columns: Topic, Words, Frequency, Timestamp. Filtered to the
            kept topics (no -1).
        """
        self._check_fitted()
        ts = list(pd.Series(timestamps).values)
        if len(ts) != len(self._documents):
            raise ValueError(
                f"timestamps has length {len(ts)} but documents has "
                f"{len(self._documents)}."
            )
        tot = self.model.topics_over_time(
            self._documents,
            ts,
            nr_bins=nr_bins,
            evolution_tuning=evolution_tuning,
        )
        return tot[tot["Topic"] != -1].reset_index(drop=True)

    def __repr__(self):
        if not self._fitted:
            return f"BERTopicModel(target={self._target_n_topics}, fitted=False)"
        return (
            f"BERTopicModel(n_topics={self.n_topics}, "
            f"n_outliers={self.n_outliers()}, fitted=True)"
        )
