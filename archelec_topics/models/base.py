"""Abstract base class for topic models.

(The style of the modelization, with one abstract class and a concrete implementation
for each model type, is inspired from the topic modelling infrastructure
I developped during my internship at the European Central Bank in 2025, 
but simplified and adapted to the needs of this project. At the ECB, 
one added difficulty was the 4 different languages in the corpus.)

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

from abc import ABC, abstractmethod

class TopicModelBase(ABC):
    """Common interface shared by all topic models in this project.
    
    All concrete models must implement fit, get_doc_topic_matrix and
    get_top_words. Methods that vary across model families (perplexity,
    reconstruction error, temporal evolution) are exposed only on the
    relevant subclass.
    """

    def __init__(self, n_topics, random_state=42):
        self.n_topics = n_topics
        self.random_state = random_state
        self._fitted = False

    @abstractmethod
    def fit(self, documents, **kwargs):
        """Train the model on a corpus. Returns self for chaining."""

    @abstractmethod
    def get_doc_topic_matrix(self):
        """Return an (n_docs, n_topics) array of topic proportions per document."""

    @abstractmethod
    def get_top_words(self, n_words=10):
        """Return a list of n_topics lists, each containing the top words of a topic."""

    def assign_dominant_topic(self):
        """Return a 1D array (n_docs,) with the argmax topic per document."""
        self._check_fitted()
        return self.get_doc_topic_matrix().argmax(axis=1)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted yet.")

    def __repr__(self):
        return f"{self.__class__.__name__}(n_topics={self.n_topics}, fitted={self._fitted})"