# Topics of French Electoral Manifestos (1973–1993) and their Temporal Evolution through the lens of NLP

**Author:** Emma Leguay — ENSAE Paris, 3A / MiE M2
**Course:** *Machine Learning for Natural Language Processing* — final project
**Track:** Topic Modelling & Temporal Evolution on the [Archelec corpus](https://archelec.sciencespo.fr/explorer)

---

## 1. Project overview

This repository contains the code, the analysis notebook and the final report of my project on the Archelec corpus of French legislative *professions de foi*. The goal is to recover the thematic content of these manifestos and its evolution over five legislative elections (1973, 1978, 1981, 1988, 1993), by crossing topic-modelling outputs with the candidate-level metadata (party, political family, occupation, sex, year).

I compare three topic models that span the practical history of the field — **LDA** (probabilistic bag-of-words), **NMF** (algebraic bag-of-words on TF-IDF) and **BERTopic** (Sentence-BERT embeddings + HDBSCAN clustering + c-TF-IDF) — and use their convergences as a triangulation device: a topic that appears in all three models is more trustworthy than one that appears in only one. NMF is used as the reference for cross-tabulations with metadata (cleanest topics on bag-of-words metrics), and BERTopic is used for the temporal-evolution analysis (finer semantic granularity).

On top of standard prevalence-per-year curves, I implement a **shift-share decomposition** that separates changes in topic prevalence into a *composition* effect (which families speak louder?) and an *intensity* effect (each family shifting its own discourse), as well as a **Jensen-Shannon distance** between successive elections to measure the amplitude of discursive recomposition.

The full report — methodology, model choices, results and limitations — is in [`LEGUAY_ML_for_NLP_project.pdf`](./LEGUAY_ML_for_NLP_project.pdf). The headline findings are summarised in §6 below.

---

## 2. Repository layout

```
.
├── README.md                              # this file
├── requirements.txt                       # pinned Python dependencies
├── LEGUAY_ML_for_NLP_project.pdf          # the final NeurIPS-style report
├── main_analysis.ipynb                    # notebook driving the whole analysis
│
├── archelec_topics/                       # Python package with the pipeline
│   ├── __init__.py
│   ├── loader.py                          # join OCR text files + metadata CSV
│   ├── description.py                     # descriptive statistics helpers
│   ├── preprocessing.py                   # OCR cleanup, n-grams, lemmatisation, stopwords
│   ├── parties_processing.py              # raw party labels → canonical parties → families
│   ├── occupations_processing.py          # raw occupations → categories
│   ├── evaluation.py                      # Cv coherence + topic diversity, K-sweep
│   ├── interpretation.py                  # top words, representative docs, topic alignment
│   ├── visualisation.py                   # NeurIPS-styled plots (word clouds, heatmaps, …)
│   ├── topic_metadata_analysis.py         # topics × {party, family, occupation} z-scores
│   ├── temporal_evolution_analysis.py     # shift-share, JSD, prevalence over time
│   └── models/
│       ├── __init__.py
│       ├── base.py                        # TopicModelBase abstract class (common API)
│       ├── lda_model.py                   # Gensim LDA wrapper
│       ├── nmf_model.py                   # scikit-learn NMF wrapper
│       └── bertopic_model.py              # BERTopic wrapper + embedding helper
│
└── data/                                  # NOT versioned — see §4 below
    ├── archelect_search.csv               # metadata exported from the Archelec explorer
    ├── arkindex_archelec/text_files/      # OCR'ed text files (one .txt per manifesto)
    └── processed/                         # cached tokens, embeddings, model outputs
```

The design of the modelling subpackage — one abstract base class plus one concrete wrapper per model family with a common API for top words, document-topic matrix and dominant topic — is inspired by the topic-modelling infrastructure I worked on during my internship at the European Central Bank in 2025, simplified to the present project's scope.

---

## 3. Setup

### 3.1 Python environment

The code was developed and tested with **Python 3.11**. The steps below assume a fresh virtual environment.

```bash
# Clone the repo and enter it
git clone <repo-url>
cd <repo-name>

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate         # on macOS / Linux
# .venv\Scripts\activate          # on Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download the spaCy French model used by the preprocessing pipeline
python -m spacy download fr_core_news_md
```

### 3.2 GPU note for BERTopic (optional but strongly recommended)

The BERTopic step embeds the ~21k manifestos with a long-context multilingual transformer (`BAAI/bge-m3` at `max_seq_length=2048`). On a CPU this takes many hours; on a Colab T4 GPU it runs in roughly two hours. The embeddings are pickled to `data/processed/bge_m3_2048_embeddings.pkl` and the notebook reloads them from disk on subsequent runs. If you do not need to recompute the embeddings, no GPU is required to run the analysis end-to-end.

---

## 4. Data

The corpus is **not** redistributed in this repository (size + licensing). It must be downloaded from its two original sources and placed under `data/`:

1. **Metadata CSV** — exported from the Archelec Sciences Po explorer at https://archelec.sciencespo.fr/explorer. The notebook expects it at `data/archelect_search.csv`.
2. **OCR'ed transcriptions** — one `.txt` file per manifesto, available in the Arkindex/Teklia repository at https://gitlab.teklia.com/ckermorvant/arkindex_archelec. The notebook expects them under `data/arkindex_archelec/text_files/` (recursively searched).

After running the loading and preprocessing steps once with `FIRST_RUN = True` in the notebook, intermediate artefacts (lemmatised tokens, embeddings, fitted models) are pickled under `data/processed/` and reloaded on subsequent runs (`FIRST_RUN = False`), so the heavy steps only happen once.

After the join with metadata and after the quality filter (≥ 500 raw characters, between 50 and 1000 useful tokens), the analysis sample contains about 21,167 manifestos covering five legislative elections.

---

## 5. How to run the analysis

The whole analysis is driven by **`main_analysis.ipynb`**, which calls the `archelec_topics` package step by step. Open the notebook in Jupyter / VS Code / Colab and execute the cells in order. The structure mirrors the report:

1. **Loading and descriptive statistics** — corpus join, metadata diagnostics, candidate sociology.
2. **Preprocessing** — OCR cleanup, acronym expansion, hand-curated n-grams, spaCy lemmatisation + POS filter, stopword removal, quality filter.
3. **LDA and NMF** — K-sweep with Cv coherence and topic diversity, fit at K=9, manual labelling, cross-model alignment.
4. **BERTopic** — embedding with `BAAI/bge-m3` at 2048 tokens, HDBSCAN clustering, outlier reduction by embedding similarity, hierarchical reduction to 25 topics, manual labelling.
5. **Topics × metadata** — per-topic z-scores by political family, party, occupation, occupation category.
6. **Temporal evolution** — prevalence per year, shift-share decomposition (composition vs intensity, both globally and per consecutive election transition), Jensen-Shannon distance between successive elections.

The first run is the long one (preprocessing ~10–15 min on CPU, embeddings ~2 h on a T4 GPU). All subsequent runs reuse the pickled artefacts and complete in a few minutes.

---

## 6. Headline findings

Within the limits of an OCR'ed corpus and of unsupervised topic modelling, the analysis supports five substantive conclusions, all detailed in §7 of the report:

1. The discursive space of French legislative *professions de foi* between 1973 and 1993 is **well stratified by political family**, with the strongest specialisation on the radical-left and far-right ends and a flatter signature in the centre and the moderate right.
2. The radical fringes do not speak in a single register but in **a small bundle of organisationally distinct sub-registers** that BERTopic recovers and bag-of-words models tend to collapse — three FN registers, three radical-left registers, two ecology registers.
3. The post-1981 decline of the *Programme commun* / class-struggle vocabulary is, in this corpus, an **intensity effect concentrated in the 1981→1988 transition**: the historically left-wing families remain present, but they speak about other things (the *tournant de la rigueur* of 1983 made quantitatively visible).
4. The temporal recomposition of the discursive space, measured by the Jensen-Shannon distance between successive elections, **peaks in 1981→1988 — not in 1978→1981** — and this ordering is robust across the three models. The discursive *alternance* lags the political one by a cycle.
5. The rise of the *crisis / immigration / insecurity* register is a **composition effect in 1981→1988** (the FN appears as a measurable share of the corpus and brings its own register) and an **intensity effect in 1988→1993** (the same register diffuses to other families), consistent with the political-science literature on the diffusion of FN-coined frames into mainstream discourse.

The shift-share decomposition applied per consecutive election transition (rather than globally over 1973–1993) is the analytical step that makes points 3 to 5 visible; a single-period decomposition would average out the chronology.

---

## 7. Limitations and possible extensions

The report (§7) discusses six limitations and three natural model extensions. The most important ones, briefly:

- **OCR quality and document length** drop noticeably between 1978 and 1988. Whether this is a genuine stylistic shift or a transcription artefact is not fully resolved here — fitting topic models on length-stratified subsamples would help control for it.
- **Profession metadata** is missing for ~44% of candidates, which restricts the topic × profession analysis to roughly half of the corpus.
- **LDA Dirichlet hyperparameters** are estimated with `'auto'` (asymmetric prior) but not grid-searched. A sweep over (α, η) ∈ {0.01, 0.1, 0.5}² would likely close part of the LDA / NMF readability gap.
- **BERTopic** uses hard cluster assignments, which mechanically inflates the absolute JSD relative to soft-assignment models. Running `approximate_distribution` on the BERTopic output would harmonise the absolute levels (at non-trivial computational cost).

Three natural model extensions, in increasing order of effort: the **Dynamic Topic Model** (Blei & Lafferty, 2006) for natively temporal vocabulary drift; the **Embedded Topic Model** (Dieng et al., 2020) to combine LDA's probabilistic structure with embedding-based semantics; and the **Structural Topic Model** (Roberts et al., 2019), which would let the candidate metadata enter the topic model itself as covariates.

---

## 8. Citation and acknowledgements

If you reuse this code or the analysis, please cite the report and link back to this repository.

The modelling infrastructure (abstract base class + one wrapper per model family, common API for top words / doc-topic matrix / dominant topic, K-sweep + coherence/diversity evaluation, shift-share decomposition) is inspired by — but simpler than — the topic-modelling pipeline I worked on at the European Central Bank in 2025. All preprocessing decisions, design and modelling choices, code, results and interpretations in this repository are mine.

I used a generative AI assistant (Claude) for help with (i) the descriptive-statistics helpers, (ii) the matplotlib styling to align figures with the NeurIPS template, (iii) understanding common OCR error patterns and refining the preprocessing accordingly, (iv) normalising the raw party-support and occupation strings into the canonical mappings used in `parties_processing.py` and `occupations_processing.py`, and (v) light proofreading of the English of the report.

Many thanks to Mr. Christopher Kermorvant for making such an interesting corpus available, and for the *Machine Learning for NLP* course that motivated this analysis.

---

## 9. Key references

- Blei, Ng & Jordan (2003), *Latent Dirichlet Allocation*, JMLR.
- Lee & Seung (1999), *Learning the parts of objects by non-negative matrix factorization*, Nature.
- Grootendorst (2022), *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*, arXiv:2203.05794.
- Reimers & Gurevych (2019), *Sentence-BERT: Sentence embeddings using Siamese BERT-Networks*, EMNLP-IJCNLP.
- McInnes, Healy & Melville (2018), *UMAP: Uniform Manifold Approximation and Projection*, arXiv:1802.03426.
- Campello, Moulavi & Sander (2013), *Density-based clustering based on hierarchical density estimates* (HDBSCAN), PAKDD.
- Röder, Both & Hinneburg (2015), *Exploring the space of topic coherence measures* (Cv), WSDM.
- Dieng, Ruiz & Blei (2020), *Topic Modeling in Embedding Spaces* (ETM), TACL.
- Gaultier-Voituriez (2022), *Archelec, les archives électorales françaises de la Ve République*.

The full bibliography is in the report.
