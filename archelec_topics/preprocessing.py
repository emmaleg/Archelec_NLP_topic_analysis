"""Text preprocessing for the Archelec corpus.

Pipeline (like in the course slides): OCR cleanup -> spaCy lemmatize + POS filter -> stopword removal
Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

import re
import unicodedata

import spacy
from tqdm import tqdm


# Domain-specific stopwords on top of spaCy's defaults.

# The list of extra stopwords was modified iteratively by looking at the most common tokens 
# in the corpus and spotting which ones were mostly noise for topic modeling. 
EXTRA_STOPWORDS = {
    "français", "française", "francais", "francaise",
    "candidat", "candidate", "candidats",
    "élection", "election", "élections", "elections", "scrutin", "scrutins",
    "circonscription", "député", "deputee", "depute", "deputes",
    "voter", "vote", "votez", "voix",
    "monsieur", "madame", "messieurs", "mesdames",
    "cher", "chère", "chers", "chères",
    "janvier", "fevrier", "février", "mars", "avril", "mai", "juin",
    "juillet", "aout", "août", "septembre", "octobre", "novembre",
    "decembre", "décembre",
}

_ARCHIVAL_PATTERNS = [
    r"sciences\s*po\s*[/\\\-]*\s*fonds\s*cevipof",
    r"r[ée]publique\s+fran[cç]aise",
    r"[ée]lections?\s+l[ée]gislatives?",
]
_ARCHIVAL_RE = re.compile("|".join(_ARCHIVAL_PATTERNS), flags=re.IGNORECASE)


def clean_ocr(text):
    """Light OCR cleanup before lemmatization.
    
    Steps:
        1. Remove recurring archival headers (Sciences Po, République française, 
           Élections législatives — they appear in most documents)
        2. Mend hyphenated line breaks
        3. Normalize Unicode
        4. Lowercase, drop non-letter characters and isolated single chars
        5. Collapse repeated whitespace
    """
    text = _ARCHIVAL_RE.sub(" ", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", " ", text)
    text = re.sub(r"\b\w\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_corpus(
    texts,
    spacy_model="fr_core_news_md",
    keep_pos=("NOUN", "PROPN", "ADJ", "VERB"),
    min_token_len=3,
):
    """Lemmatize and filter a list of raw texts.
    
    Returns a list of lists of lemmas, ready for topic modeling.
    """
    nlp = spacy.load(spacy_model, disable=["ner", "parser"])
    stopwords = set(nlp.Defaults.stop_words) | EXTRA_STOPWORDS
    keep_pos_set = set(keep_pos)
    
    cleaned = [clean_ocr(t) for t in texts]
    
    out = []
    for doc in tqdm(nlp.pipe(cleaned, batch_size=64), total=len(cleaned), desc="Lemmatizing"):
        tokens = [
            tok.lemma_.lower()
            for tok in doc
            if (
                tok.pos_ in keep_pos_set
                and tok.lemma_.lower() not in stopwords
                and len(tok.lemma_) >= min_token_len
                and tok.lemma_.isalpha()
            )
        ]
        out.append(tokens)
    return out