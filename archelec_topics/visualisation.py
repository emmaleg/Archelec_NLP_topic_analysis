"""Visualization functions for topic models.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

import matplotlib.pyplot as plt

def plot_topic_words(top_words_per_topic, title, n_cols=5):
    n_topics = len(top_words_per_topic)
    n_rows = (n_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    for k, words in enumerate(top_words_per_topic):
        ax = axes[k]
        n = len(words)
        ax.barh(range(n), range(n, 0, -1), color="#332288")
        ax.set_yticks(range(n))
        ax.set_yticklabels(words, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Topic {k}", fontsize=10)
        ax.set_xticks([])
    for k in range(n_topics, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig