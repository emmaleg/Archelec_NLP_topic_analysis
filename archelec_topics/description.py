"""Statistical description of the Archelec corpus.

These helpers will help me feed the corpus description section of the report
and let the reader judge data quality before any modeling.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

from __future__ import annotations

import pandas as pd

def column_overview(df, exclude=("text",)) -> pd.DataFrame:
    """For each column, return its dtype, non-null count, and number of unique values.
    
    Useful to spot which columns are interesting to analyze further.
    """
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        rows.append({
            "column":     col,
            "dtype":      str(df[col].dtype),
            "n_non_null": df[col].notna().sum(),
            "n_unique":   df[col].nunique(),
            "example":    df[col].dropna().iloc[0] if df[col].notna().any() else None,
        })
    return pd.DataFrame(rows)

def value_distribution(df, col, top_n=20) -> pd.DataFrame:
    """Return value counts for a column, including NaN, with optional proportions.
    
    Returns a DataFrame with absolute counts and proportions, easier
    to read than the raw Series.
    """
    counts = df[col].value_counts(dropna=False).head(top_n)
    out = pd.DataFrame({
        "n_documents": counts,
        "proportion":  counts / len(df),
    })
    out.index.name = col
    return out

def text_statistics(df, text_col="text") -> pd.DataFrame:
    """Summary statistics on document length, in chars and words.
    
    Returns a DataFrame with total, mean, median, std, min, max
    for both characters and words.
    """
    char_lengths = df[text_col].str.len()
    word_lengths = df[text_col].str.split().str.len()
    
    stats = pd.DataFrame({
        "n_chars": char_lengths,
        "n_words": word_lengths,
    })
    
    summary = pd.DataFrame({
        "total":  stats.sum(),
        "mean":   stats.mean(),
        "median": stats.median(),
        "std":    stats.std(),
        "min":    stats.min(),
        "max":    stats.max(),
    })
    return summary

def temporal_distribution(df, group_col, year_col="year", top_n=None) -> pd.DataFrame:
    """Cross-tabulate documents by year and any categorical column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain `year_col` and `group_col`.
    group_col : str
        The column used as pivot (e.g., 'titulaire-soutien', 'titulaire-liste').
    year_col : str
        Column containing the year. Defaults to 'year'.
    top_n : int, optional
        If set, keep only the top N most frequent values of `group_col`
        (across all years) and group the rest into 'OTHER'.
    
    Returns
    -------
    pd.DataFrame
        Years as rows, group values as columns, counts as values.
    """
    sub = df.dropna(subset=[year_col, group_col]).copy()
    
    if top_n is not None:
        top_values = sub[group_col].value_counts().head(top_n).index
        sub.loc[~sub[group_col].isin(top_values), group_col] = "OTHER"
    
    pivot = (
        sub.groupby([year_col, group_col])
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )
    pivot.index = pivot.index.astype(int)
    return pivot