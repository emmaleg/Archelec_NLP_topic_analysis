"""Latent Dirichlet Allocation via Gensim.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

import numpy as np
from archelec_topics.models.base import TopicModelBase


class LDATopicModel(TopicModelBase):
    """Wrapper around gensim.models.LdaModel.
    
    Parameters
    ----------
    n_topics : int
    no_below : int
        Drop tokens occurring in fewer than no_below documents.
    no_above : float
        Drop tokens occurring in more than no_above fraction of documents.
    keep_n : int
        Cap the vocabulary at the keep_n most frequent tokens.
    passes : int
        Number of training passes over the corpus.
    iterations : int
        Maximum inference iterations per document.
    alpha, eta : str or float
        Dirichlet hyperpriors. 'auto' lets gensim learn them.
    random_state : int
    """

    def __init__(
        self,
        n_topics=15,
        no_below=10,
        no_above=0.5,
        keep_n=8000,
        passes=10,
        iterations=50,
        alpha="auto",
        eta="auto",
        random_state=42,
    ):
        super().__init__(n_topics=n_topics, random_state=random_state)
        self.no_below   = no_below
        self.no_above   = no_above
        self.keep_n     = keep_n
        self.passes     = passes
        self.iterations = iterations
        self.alpha      = alpha
        self.eta        = eta

        self.model       = None
        self.dictionary  = None
        self.corpus      = []

    def fit(self, documents, **kwargs):
        """Train LDA on a list of tokenized documents (list of list of str)."""
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel

        self.dictionary = Dictionary(documents)
        self.dictionary.filter_extremes(
            no_below=self.no_below,
            no_above=self.no_above,
            keep_n=self.keep_n,
        )
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            passes=self.passes,
            iterations=self.iterations,
            alpha=self.alpha,
            eta=self.eta,
            random_state=self.random_state,
        )
        self._fitted = True
        return self

    def get_doc_topic_matrix(self):
        """Return (n_docs, n_topics) probability matrix."""
        self._check_fitted()
        out = np.zeros((len(self.corpus), self.n_topics), dtype=float)
        for i, bow in enumerate(self.corpus):
            for topic_id, prob in self.model.get_document_topics(bow, minimum_probability=0.0):
                out[i, topic_id] = prob
        return out

    def get_top_words(self, n_words=10):
        """Return list of n_topics lists, each with the top n_words."""
        self._check_fitted()
        return [
            [w for w, _ in self.model.show_topic(k, topn=n_words)]
            for k in range(self.n_topics)
        ]

    def perplexity(self):
        """Per-word perplexity (lower is better). LDA-specific."""
        self._check_fitted()
        log_p = self.model.log_perplexity(self.corpus)
        return float(np.exp(-log_p))