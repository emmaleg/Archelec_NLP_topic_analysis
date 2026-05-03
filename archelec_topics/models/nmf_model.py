"""Nonnegative Matrix Factorization via scikit-learn with TF-IDF input."""

import numpy as np
from archelec_topics.models.base import TopicModelBase


class NMFTopicModel(TopicModelBase):
    """Wrapper around sklearn.decomposition.NMF with TF-IDF features.
    
    Pipeline: TfidfVectorizer -> NMF
    
    Parameters
    ----------
    n_topics : int
    max_features : int or None
        Vocabulary size cap.
    min_df : int
        Min document frequency.
    max_df : float
        Max document frequency.
    init : str
        NMF initialization ('nndsvda' is good for sparse data).
    max_iter : int
    random_state : int
    """

    def __init__(
        self,
        n_topics=15,
        max_features=8000,
        min_df=10,
        max_df=0.5,
        init="nndsvda",
        max_iter=400,
        random_state=42,
    ):
        super().__init__(n_topics=n_topics, random_state=random_state)
        self.max_features = max_features
        self.min_df       = min_df
        self.max_df       = max_df
        self.init         = init
        self.max_iter     = max_iter

        self.vectorizer     = None
        self.model          = None
        self.feature_names_ = None
        self._W = None  # doc-topic
        self._H = None  # topic-word

    def fit(self, documents, **kwargs):
        """Fit on tokenized documents (list of list of str). Joins to space-separated strings internally."""
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer

        # NMF needs strings, not token lists
        if documents and isinstance(documents[0], list):
            docs = [" ".join(d) for d in documents]
        else:
            docs = list(documents)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=False,                 # already lowercased upstream
            token_pattern=r"(?u)\b\w{2,}\b",
        )
        X = self.vectorizer.fit_transform(docs)
        self.feature_names_ = np.array(self.vectorizer.get_feature_names_out())

        self.model = NMF(
            n_components=self.n_topics,
            init=self.init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._W = self.model.fit_transform(X)
        self._H = self.model.components_
        self._fitted = True
        return self

    def get_doc_topic_matrix(self):
        """Return (n_docs, n_topics) matrix, normalized to probabilities."""
        self._check_fitted()
        row_sums = self._W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return self._W / row_sums

    def get_top_words(self, n_words=10):
        """Top n_words per topic, ranked by NMF component weight."""
        self._check_fitted()
        return [
            self.feature_names_[self._H[k].argsort()[::-1][:n_words]].tolist()
            for k in range(self.n_topics)
        ]

    def reconstruction_err(self):
        """Frobenius reconstruction error (lower is better). NMF-specific."""
        self._check_fitted()
        return float(self.model.reconstruction_err_)