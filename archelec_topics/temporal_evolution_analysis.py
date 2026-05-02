"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


def build_tyk_tensor(doc_topic, family_dummies, years, topic_labels):
    """Construit le tenseur P[t, f, k] et la matrice w[t, f].

    Parameters
    ----------
    doc_topic : (n_docs, n_topics) array
        Matrice doc × topic (poids ou one-hot pour BERTopic).
    family_dummies : (n_docs, n_families) DataFrame
        Indicatrices de famille politique (lignes alignées sur doc_topic).
    years : (n_docs,) array-like
        Année de chaque document.
    topic_labels : list[str]
        Labels lisibles des topics, dans l'ordre des colonnes de doc_topic.

    Returns
    -------
    P    : xarray-like dict {(year, family) -> np.ndarray of size n_topics}
           encodé comme DataFrame multi-index (year, family) × topic.
    w    : DataFrame (year × family), part de chaque famille dans le corpus.
    """
    df_local = pd.DataFrame(doc_topic, columns=topic_labels)
    df_local["year"]   = np.asarray(years)
    fam_cols           = list(family_dummies.columns)
    df_local[fam_cols] = family_dummies.values

    # Famille principale d'un document = celle dont le poids est non nul.
    # On utilise un argmax sur les dummies (les docs sans famille reçoivent
    # NaN et sont écartés).
    fam_mat = family_dummies.values
    has_fam = fam_mat.sum(axis=1) > 0
    primary = np.where(has_fam, fam_mat.argmax(axis=1), -1)
    df_local["family"] = [fam_cols[i] if i >= 0 else None for i in primary]
    df_local = df_local.dropna(subset=["family"])

    # P[t, f, k] : poids moyen du topic k dans (t, f)
    P = (df_local
         .groupby(["year", "family"])[topic_labels]
         .mean()
         .sort_index())

    # w[t, f] : part de la famille f dans les docs de l'année t
    counts = (df_local
              .groupby(["year", "family"])
              .size()
              .unstack(fill_value=0))
    w = counts.div(counts.sum(axis=1), axis=0)

    return P, w


def shift_share_decomposition(P, w, year_start, year_end):
    """Décompose ΔP_global[k] en effet composition + effet intensité.

    Returns
    -------
    DataFrame avec colonnes :
        topic, P_start, P_end, delta_total,
        effect_composition, effect_intensity, residual
    """
    topic_labels = P.columns.tolist()
    families     = w.columns.tolist()

    # On restreint au support commun (familles présentes aux deux dates)
    P_t0 = P.xs(year_start, level="year").reindex(families).fillna(0.0)
    P_t1 = P.xs(year_end,   level="year").reindex(families).fillna(0.0)
    w_t0 = w.loc[year_start].reindex(families).fillna(0.0)
    w_t1 = w.loc[year_end  ].reindex(families).fillna(0.0)

    # Moyennes pour la décomposition « symétrique » (à la Esteban)
    P_mean = (P_t0 + P_t1) / 2.0
    w_mean = (w_t0 + w_t1) / 2.0

    delta_w = w_t1 - w_t0           # Series indexée par famille
    delta_P = P_t1 - P_t0           # DataFrame famille × topic

    P_global_t0 = (w_t0.values[:, None] * P_t0.values).sum(axis=0)
    P_global_t1 = (w_t1.values[:, None] * P_t1.values).sum(axis=0)
    delta_total = P_global_t1 - P_global_t0

    eff_composition = (delta_w.values[:, None] * P_mean.values).sum(axis=0)
    eff_intensity   = (w_mean.values[:, None]  * delta_P.values).sum(axis=0)

    residual = delta_total - (eff_composition + eff_intensity)

    out = pd.DataFrame({
        "topic"             : topic_labels,
        "P_start"           : P_global_t0,
        "P_end"             : P_global_t1,
        "delta_total"       : delta_total,
        "effect_composition": eff_composition,
        "effect_intensity"  : eff_intensity,
        "residual"          : residual,
    }).set_index("topic")
    return out


def plot_shift_share(decomp, model_name, year_start, year_end, ax=None):
    """Bar chart : pour chaque topic, contribution de chaque effet à ΔP."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(decomp) + 1.5))

    ordered = decomp.sort_values("delta_total")
    y       = np.arange(len(ordered))
    h       = 0.4

    ax.barh(y - h/2, ordered["effect_composition"], height=h,
            label="Composition (qui parle ?)", color="#882255")
    ax.barh(y + h/2, ordered["effect_intensity"],  height=h,
            label="Intensité (combien on en parle ?)", color="#117733")
    ax.scatter(ordered["delta_total"], y, color="black", s=22, zorder=3,
               label="Δ total")

    ax.set_yticks(y)
    ax.set_yticklabels(ordered.index, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel(f"Variation de prévalence ({year_start} → {year_end})")
    ax.set_title(f"{model_name} — Shift-share decomposition")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_topic_trajectories_by_family(P, w, model_name, top_n_topics=8,
                                       top_n_families=6, min_share=0.02):
    """Small multiples : pour chaque topic, courbes par famille politique.

    Parameters
    ----------
    P, w : sortie de build_tyk_tensor.
    top_n_topics : int
        On affiche les topics dont la prévalence globale moyenne est la plus
        élevée (les autres encombreraient la lecture).
    top_n_families : int
        Idem côté familles.
    min_share : float
        Seuil sous lequel on n'affiche pas un point (famille trop peu
        représentée à cette élection-là — courbe trop bruitée).
    """
    topic_labels = P.columns.tolist()
    families     = w.columns.tolist()
    years        = sorted(w.index.tolist())

    # Topics les plus prévalents en moyenne (corpus entier)
    P_global = (P
                .multiply(w.stack(future_stack=True), axis=0)
                .groupby(level="year").sum()
                .mean(axis=0))
    top_topics = P_global.nlargest(top_n_topics).index.tolist()

    # Familles les plus présentes en moyenne
    top_fams = w.mean(axis=0).nlargest(top_n_families).index.tolist()

    n_cols = 4
    n_rows = int(np.ceil(len(top_topics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 2.6 * n_rows),
                             sharex=True)
    axes = np.atleast_2d(axes).flatten()

    for ax, topic in zip(axes, top_topics):
        for fam in top_fams:
            xs, ys = [], []
            for y in years:
                if (y, fam) not in P.index:
                    continue
                if w.loc[y, fam] < min_share:
                    continue            # famille marginale → on saute
                xs.append(y)
                ys.append(P.loc[(y, fam), topic])
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.1, markersize=3,
                        label=fam)
        ax.set_title(topic, fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes[len(top_topics):]:
        ax.axis("off")

    # Légende globale en bas
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(top_fams), 6), fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"{model_name} — Trajectoire des topics par famille politique",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    return fig


def topic_concentration_by_family(P, w):
    """Entropie (en bits) de la distribution de topics moyenne, par (année, famille).
    Faible entropie = discours mono-thématique ; haute = diversifié."""
    rows = []
    for (year, fam), profile in P.iterrows():
        if w.loc[year, fam] < 0.01:
            continue
        p = profile.values.astype(float)
        s = p.sum()
        if s <= 0:
            continue
        p = p / s
        # entropie en bits, ignorer les 0
        H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        rows.append({"year": year, "family": fam, "entropy": H})
    return pd.DataFrame(rows).pivot(index="family", columns="year",
                                     values="entropy")


def jsd_between_elections(P, w):
    """Divergence Jensen-Shannon entre la distribution de topics moyenne
    de deux élections successives. Mesure l'amplitude de la recomposition
    discursive globale."""
    years = sorted(w.index.tolist())
    P_global = pd.DataFrame(index=years, columns=P.columns, dtype=float)
    for y in years:
        sub_P = P.xs(y, level="year")
        sub_w = w.loc[y].reindex(sub_P.index).fillna(0.0)
        P_global.loc[y] = (sub_w.values[:, None] * sub_P.values).sum(axis=0)
    # Normalisation pour obtenir une distribution de probabilité
    P_global = P_global.div(P_global.sum(axis=1), axis=0)

    rows = []
    for y0, y1 in zip(years[:-1], years[1:]):
        d = jensenshannon(P_global.loc[y0].values,
                          P_global.loc[y1].values, base=2)
        rows.append({"transition": f"{y0}→{y1}", "JSD": d})
    return pd.DataFrame(rows)