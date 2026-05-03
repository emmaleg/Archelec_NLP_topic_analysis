"""Evaluation metrics for topic models."""

import pandas as pd
from tqdm import tqdm


def compute_cv_coherence(top_words_per_topic, tokenized_texts, dictionary=None, n_words=10):
    """Compute Cv coherence (Röder et al. 2015) via Gensim.
    
    Higher is better, range roughly [0, 1]. The metric most correlated
    with human judgments of topic interpretability.
    """
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel

    if dictionary is None:
        dictionary = Dictionary(tokenized_texts)

    vocab = set(dictionary.token2id)
    truncated = [
        [w for w in topic[:n_words] if w in vocab]
        for topic in top_words_per_topic
    ]
    truncated = [t for t in truncated if len(t) >= 2]
    if not truncated:
        return float("nan")

    cm = CoherenceModel(
        topics=truncated,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(cm.get_coherence())


def compute_topic_diversity(top_words_per_topic, n_words=25):
    """Topic diversity: proportion of unique words across all topics' top-N.
    
    Range [0, 1], higher is better. 1.0 = topics are completely distinct.
    """
    if not top_words_per_topic:
        return float("nan")
    seen = []
    for topic in top_words_per_topic:
        seen.extend(topic[:n_words])
    if not seen:
        return float("nan")
    return len(set(seen)) / len(seen)


def summarize_model(model, tokenized_texts, n_words_coh=10, n_words_div=25):
    """Single-row summary of a fitted model. Returns a dict."""
    top = model.get_top_words(n_words=max(n_words_coh, n_words_div))
    out = {
        "model":        model.__class__.__name__,
        "n_topics":     model.n_topics,
        "coherence_cv": compute_cv_coherence(top, tokenized_texts, n_words=n_words_coh),
        "diversity":    compute_topic_diversity(top, n_words=n_words_div),
    }
    if hasattr(model, "perplexity"):
        try:
            out["perplexity"] = model.perplexity()
        except Exception:
            pass
    if hasattr(model, "reconstruction_err"):
        try:
            out["reconstruction_err"] = model.reconstruction_err()
        except Exception:
            pass
    return out


def sweep_n_topics(model_factory, n_topics_grid, documents, tokenized_texts):
    """Train one model per K in the grid and return a comparison DataFrame.
    
    Parameters
    ----------
    model_factory : callable
        f(n_topics) -> instance of TopicModelBase, not yet fitted.
    n_topics_grid : iterable of int
    documents : list
        Whatever the model expects in fit() (tokens for LDA, strings for NMF if you prefer).
    tokenized_texts : list of list of str
        Used for coherence computation (always tokenized).
    """
    rows = []
    for k in tqdm(list(n_topics_grid), desc="Sweeping K"):
        model = model_factory(k)
        model.fit(documents)
        rows.append(summarize_model(model, tokenized_texts))
    return pd.DataFrame(rows)