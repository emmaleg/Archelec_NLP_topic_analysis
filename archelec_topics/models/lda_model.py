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
    update_every : int
        Update model every update_every documents (0 for batch).
    chunksize : int
        Number of documents per training chunk (only if update_every > 0).
    alpha, eta : str or float
        Dirichlet hyperpriors. 'auto' lets gensim learn them.
    minimum_probability : float
        Minimum topic probability for a document (used in get_doc_topic_matrix).
    random_state : int
    """

    def __init__(
        self,
        n_topics=15,
        no_below=10,
        no_above=0.65,
        keep_n=18000,
        passes=10,               # Number of complete passes through the entire corpus during training. More passes can lead to better convergence, but increase training time.
        iterations=50,           # Maximum number of iterations when inferring the topic distribution of a new document. More iterations can lead to better inference, but increase time.
        update_every=1,          # Mode "online" number of chunk before updating the parameters. Setting it to 0 (or False) means batch learning, i.e. all documents are used at once to update the model parameters.
        chunksize=2000,          # Number of documents to be used in each training chunk. This is only used if update_every is not 0. A proper chunksize can speed up training significantly, especially for large corpora.
        eval_every=None,         # NEW: skip perplexity computation
        alpha="auto",            # Parameter of the concentration of topics per document : alpha big means a lot of topics per document, alpha small means few topics per document.
        eta="auto",              # Parameter of the concentration of words per topic : eta big means a lot of words per topic, eta small means few words per topic.
        minimum_probability=0.0, # Minimum probability to assign to a topic for a document (used in get_doc_topic_matrix)
        random_state=42,         # for reproductibility 
    ):
        super().__init__(n_topics=n_topics, random_state=random_state)
        self.no_below            = no_below
        self.no_above            = no_above
        self.keep_n              = keep_n
        self.passes              = passes
        self.iterations          = iterations
        self.alpha               = alpha
        self.eta                 = eta
        self.minimum_probability = minimum_probability
        self.update_every        = update_every
        self.chunksize           = chunksize
        self.eval_every          = eval_every
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
            update_every=self.update_every,
            chunksize=self.chunksize,
            eval_every=self.eval_every,
            alpha=self.alpha,
            eta=self.eta,
            minimum_probability=self.minimum_probability,
            random_state=self.random_state,
        )
        self._fitted = True
        return self

    def get_doc_topic_matrix(self):
        """Return (n_docs, n_topics) probability matrix."""
        self._check_fitted()
        out = np.zeros((len(self.corpus), self.n_topics), dtype=float)
        for i, bow in enumerate(self.corpus):
            for topic_id, prob in self.model.get_document_topics(bow, minimum_probability=self.minimum_probability):
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