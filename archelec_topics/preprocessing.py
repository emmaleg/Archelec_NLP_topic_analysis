"""Text preprocessing for the Archelec corpus.

Pipeline (like in the course slides): OCR cleanup -> spaCy lemmatize + POS filter -> stopword removal
Inspiration from my ECB work for all the custom cleaning rules, the n-grams compounding, and the stopword list.
Help of AI to identify common OCR errors, and putting together an almost exhaustive list of expressions from an historical perspective.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

import re
import unicodedata
import pandas as pd
from collections import Counter, defaultdict

import spacy
from tqdm import tqdm

# Note : this preprocessing was developped iteratively after several run of the LDA model,
# understanding which kind of words are polluting the topics and trying to get rid of them.
# In the end, I also decided to add tokens that are n-grams, because those expressions
# make sense only together, trying to take into account the vocabulary of the time.

# ============================================================================
# Helper: accent stripping (used both on lemmas in output and on stopwords)
# ============================================================================

def strip_accents(s):
    """Remove diacritical marks. 'écologie' -> 'ecologie', 'député' -> 'depute'.

    Used at the very end of the pipeline (after spaCy lemmatization) so that
    OCR variants like 'écologie' and 'ecologie' collapse to a single token.
    spaCy itself still sees the accented input, which preserves lemmatization
    quality for the French model.
    """
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if unicodedata.category(c) != "Mn"
    )

# Note : this preprocessing was developped iteratively after several run of the LDA model,
# understanding which kind of words are polluting the topics and trying to get rid of them.
# In the end, I also decided to add tokens that are n-grams, because those expressions
# make sense only together, trying to take into account the vocabulary of the time.

# ============================================================================
# (1) German paragraph filter — Alsace trilingual professions de foi
# ============================================================================
# I noticed that some professions de foi from Alsace region are both in French 
# and German, we keep only the German parts by dropping paragraphs 
# with a high proportion of common German words.

_GERMAN_MARKERS = frozenset({
    "die", "der", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "und", "oder", "ist", "sind", "war", "waren", "wird", "werden", "hat",
    "haben", "wir", "sie", "ich", "nicht", "auch", "mit", "von", "zu", "zur",
    "zum", "bei", "für", "fur", "auf", "aus", "als", "sich", "nach", "über",
    "unter", "vor", "wenn", "weil", "dass", "daß", "kann", "muss", "soll",
    "schon", "noch", "doch", "aber", "denn", "diese", "dieser", "dieses",
})


def strip_german_paragraphs(text, min_chars=40, threshold=0.10):
    """Drop paragraphs where a certain proportion (threshold) of tokens are German words.

    Splits on blank lines first, falls back to single newlines if needed.
    Short paragraphs (<min_chars) are kept since detection is unreliable on them.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) == 1:
        paragraphs = text.split("\n")

    kept = []
    for p in paragraphs:
        if len(p) < min_chars:
            kept.append(p)
            continue
        words = re.findall(r"[a-zàâçéèêëîïôûùüÿñæœß]+", p.lower())
        if not words:
            kept.append(p)
            continue
        ratio = sum(1 for w in words if w in _GERMAN_MARKERS) / len(words)
        if ratio < threshold:
            kept.append(p)
    return "\n\n".join(kept)


# ============================================================================
# (2) Compound expressions — glued BEFORE spaCy so they survive as one token
# ============================================================================
# Underscores are used in the replacement so the token remains readable in topic outputs 
# (e.g. "marche_commun") and so spaCy's tokenizer keeps it as a single unit.
#
# IMPORTANT: longer expressions must come before shorter overlapping ones,
# otherwise "marche commun" would be substituted before "marche commun
# europeen" has a chance to match.

_COMPOUND_PATTERNS_RAW = [
    # ---- Coalitions, programmes, parties (long forms first) ----
    (r"ligue\s+communiste\s+r[ée]volutionnaire",            "ligue_communiste_revolutionnaire"),
    (r"mouvement\s+des\s+radicaux\s+de\s+gauche",           "mouvement_radicaux_gauche"),
    (r"rassemblement\s+pour\s+la\s+r[ée]publique",          "rassemblement_pour_republique"),
    (r"union\s+pour\s+la\s+d[ée]mocratie\s+fran[cç]aise",   "union_democratie_francaise"),
    (r"centre\s+des\s+d[ée]mocrates\s+sociaux",             "centre_democrates_sociaux"),
    (r"parti\s+communiste\s+fran[cç]ais",                   "parti_communiste_francais"),
    (r"parti\s+socialiste\s+unifi[ée]",                     "parti_socialiste_unifie"),
    (r"g[ée]n[ée]ration\s+[ée]cologie",                     "generation_ecologie"),
    (r"union\s+de\s+la\s+gauche",                           "union_de_la_gauche"),
    (r"programme\s+commun",                                 "programme_commun"),
    (r"front\s+populaire",                                  "front_populaire"),
    (r"front\s+national",                                   "front_national"),
    (r"lutte\s+ouvri[èe]re",                                "lutte_ouvriere"),
    (r"nouvelle\s+majorit[ée]",                             "nouvelle_majorite"),
    (r"majorit[ée]\s+pr[ée]sidentielle",                    "majorite_presidentielle"),
    (r"divers\s+gauche",                                    "divers_gauche"),
    (r"divers\s+droite",                                    "divers_droite"),
    (r"parti\s+r[ée]publicain",                             "parti_republicain"),

    # ---- Politicians with compound names ----
    (r"valery\s+giscard\s+d['’]?\s*estaing",                "giscard_destaing"),
    (r"giscard\s+d['’]?\s*estaing",                         "giscard_destaing"),
    (r"jean[\s\-]+marie\s+le\s+pen",                        "le_pen"),
    (r"\ble\s+pen\b",                                       "le_pen"),
    (r"\blepen\b",                                          "le_pen"),
    (r"jacques\s+chaban[\s\-]+delmas",                      "chaban_delmas"),
    (r"chaban[\s\-]+delmas",                                "chaban_delmas"),
    (r"jean[\s\-]+jacques\s+servan[\s\-]+schreiber",        "servan_schreiber"),
    (r"servan[\s\-]+schreiber",                             "servan_schreiber"),
    (r"pierre\s+mend[èe]s\s+france",                        "mendes_france"),
    (r"mend[èe]s\s+france",                                 "mendes_france"),
    (r"jean[\s\-]+pierre\s+chev[èe]nement",                 "chevenement"),
    (r"chev[èe]nement",                                     "chevenement"),
    
    # ---- Institutions and offices ----
    (r"conseil\s+[ée]conomique\s+et\s+social",              "conseil_economique_social"),
    (r"conseil\s+constitutionnel",                          "conseil_constitutionnel"),
    (r"conseil\s+d['’]\s*[ée]tat",                          "conseil_etat"),
    (r"conseil\s+des\s+ministres",                          "conseil_ministres"),
    (r"conseil\s+de\s+prud['’]?\s*hommes",                  "conseil_prudhommes"),
    (r"conseil\s+municipal",                                "conseil_municipal"),
    (r"conseil\s+g[ée]n[ée]ral",                            "conseil_general"),
    (r"conseil\s+r[ée]gional",                              "conseil_regional"),
    (r"assembl[ée]e\s+nationale",                           "assemblee_nationale"),
    (r"pr[ée]sident\s+de\s+la\s+r[ée]publique",             "president_republique"),
    (r"premier\s+ministre",                                 "premier_ministre"),
    (r"chef\s+de\s+l['’]?\s*[ée]tat",                       "chef_etat"),
    (r"garde\s+des\s+sceaux",                               "garde_sceaux"),
    (r"haute\s+administration",                             "haute_administration"),
    (r"fonction\s+publique",                                "fonction_publique"),
    (r"service\s+public",                                   "service_public"),
    (r"collectivit[ée]s\s+locales",                         "collectivites_locales"),
    (r"collectivit[ée]s\s+territoriales",                   "collectivites_territoriales"),

    # ---- Economic and social vocabulary ----
    (r"contribution\s+sociale\s+g[ée]n[ée]ralis[ée]e",      "contribution_sociale_generalisee"),
    (r"revenu\s+minimum\s+d['’]?\s*insertion",              "revenu_minimum_insertion"),
    (r"imp[ôo]t\s+sur\s+les\s+grandes\s+fortunes",          "impot_grandes_fortunes"),
    (r"imp[ôo]t\s+sur\s+la\s+fortune",                      "impot_fortune"),
    (r"petites\s+et\s+moyennes\s+entreprises",              "petites_moyennes_entreprises"),
    (r"pouvoir\s+d['’]?\s*achat",                           "pouvoir_achat"),
    (r"s[ée]curit[ée]\s+sociale",                           "securite_sociale"),
    (r"assurance\s+maladie",                                "assurance_maladie"),
    (r"assurance\s+vieillesse",                             "assurance_vieillesse"),
    (r"assurance\s+ch[ôo]mage",                             "assurance_chomage"),
    (r"allocations?\s+familiales",                          "allocations_familiales"),
    (r"minimum\s+vieillesse",                               "minimum_vieillesse"),
    (r"salaire\s+minimum",                                  "salaire_minimum"),
    (r"contrat\s+de\s+travail",                             "contrat_travail"),
    (r"code\s+du\s+travail",                                "code_travail"),
    (r"comit[ée]\s+d['’]?\s*entreprise",                    "comite_entreprise"),
    (r"partenaires\s+sociaux",                              "partenaires_sociaux"),
    (r"conventions?\s+collectives?",                        "conventions_collectives"),
    (r"classe\s+ouvri[èe]re",                               "classe_ouvriere"),
    (r"classe\s+moyenne",                                   "classe_moyenne"),
    (r"grandes\s+entreprises",                              "grandes_entreprises"),
    (r"moyens\s+de\s+production",                           "moyens_production"),
    (r"forces\s+productives",                               "forces_productives"),
    (r"relance\s+[ée]conomique",                            "relance_economique"),
    (r"plan\s+de\s+rigueur",                                "plan_rigueur"),
    (r"plan\s+barre",                                       "plan_barre"),
    (r"libre\s+entreprise",                                 "libre_entreprise"),
    (r"[ée]conomie\s+de\s+march[ée]",                       "economie_marche"),
    (r"[ée]conomie\s+sociale",                              "economie_sociale"),
    (r"d[ée]ficit\s+public",                                "deficit_public"),
    (r"dette\s+publique",                                   "dette_publique"),
    (r"am[ée]nagement\s+du\s+territoire",                   "amenagement_territoire"),
    (r"d[ée]veloppement\s+local",                           "developpement_local"),

    # ---- Inflation, monnaie, vie chère ----
    (r"vie\s+ch[èe]re",                                     "vie_chere"),
    (r"hausse\s+des\s+prix",                                "hausse_prix"),
    (r"flamb[ée]e\s+des\s+prix",                            "flambee_prix"),
    (r"baisse\s+des\s+prix",                                "baisse_prix"),
    (r"co[ûu]t\s+de\s+la\s+vie",                            "cout_vie"),
    (r"niveau\s+de\s+vie",                                  "niveau_vie"),
    (r"blocage\s+des\s+prix",                               "blocage_prix"),
    (r"contr[ôo]le\s+des\s+prix",                           "controle_prix"),
    (r"d[ée]rapage\s+des\s+prix",                           "derapage_prix"),
    (r"spirale\s+inflationniste",                           "spirale_inflationniste"),
    (r"d[ée]valuation\s+du\s+franc",                        "devaluation_franc"),
    (r"franc\s+fort",                                       "franc_fort"),
    (r"franc\s+faible",                                     "franc_faible"),
    (r"d[ée]sindexation\s+des\s+salaires",                  "desindexation_salaires"),
    (r"indexation\s+des\s+salaires",                        "indexation_salaires"),
    (r"[ée]chelle\s+mobile\s+des\s+salaires",               "echelle_mobile_salaires"),

    # ---- Energy and environment ----
    (r"[ée]nergie\s+nucl[ée]aire",                          "energie_nucleaire"),
    (r"centrales?\s+nucl[ée]aires?",                        "centrale_nucleaire"),
    (r"programme\s+nucl[ée]aire",                           "programme_nucleaire"),
    (r"choc\s+p[ée]trolier",                                "choc_petrolier"),
    (r"crise\s+p[ée]troli[èe]re",                           "crise_petroliere"),
    (r"ind[ée]pendance\s+[ée]nerg[ée]tique",                "independance_energetique"),
    (r"[ée]nergies?\s+renouvelables?",                      "energies_renouvelables"),

    # ---- Europe and international ----
    (r"communaut[ée]\s+[ée]conomique\s+europ[ée]enne",      "communaute_economique_europeenne"),
    (r"politique\s+agricole\s+commune",                     "politique_agricole_commune"),
    (r"syst[èe]me\s+mon[ée]taire\s+europ[ée]en",            "systeme_monetaire_europeen"),
    (r"acte\s+unique\s+europ[ée]en",                        "acte_unique_europeen"),
    (r"trait[ée]\s+de\s+rome",                              "traite_rome"),
    (r"trait[ée]\s+de\s+maastricht",                        "traite_maastricht"),
    (r"parlement\s+europ[ée]en",                            "parlement_europeen"),
    (r"[ée]lections?\s+europ[ée]ennes?",                    "elections_europeennes"),
    (r"communaut[ée]\s+europ[ée]enne",                      "communaute_europeenne"),
    (r"march[ée]\s+commun",                                 "marche_commun"),
    (r"monnaie\s+unique",                                   "monnaie_unique"),
    (r"europe\s+sociale",                                   "europe_sociale"),
    (r"europe\s+politique",                                 "europe_politique"),
    (r"guerre\s+froide",                                    "guerre_froide"),
    (r"guerre\s+du\s+golfe",                                "guerre_golfe"),
    (r"mur\s+de\s+berlin",                                  "mur_berlin"),
    (r"pacte\s+de\s+varsovie",                              "pacte_varsovie"),
    (r"bloc\s+sovi[ée]tique",                               "bloc_sovietique"),
    (r"tiers\s+monde",                                      "tiers_monde"),
    (r"aide\s+au\s+d[ée]veloppement",                       "aide_developpement"),
    (r"droits?\s+de\s+l['’]?\s*homme",                      "droits_homme"),
    (r"proche[\s\-]+orient",                                "proche_orient"),
    (r"moyen[\s\-]+orient",                                 "moyen_orient"),
    (r"afrique\s+du\s+nord",                                "afrique_nord"),
    (r"force\s+de\s+frappe",                                "force_frappe"),
    (r"dissuasion\s+nucl[ée]aire",                          "dissuasion_nucleaire"),
    (r"armes?\s+nucl[ée]aires?",                            "arme_nucleaire"),
    (r"bombe\s+atomique",                                   "bombe_atomique"),
    (r"non[\s\-]+prolif[ée]ration",                         "non_proliferation"),

    # ---- Immigration, identity, citizenship ----
    (r"pr[ée]f[ée]rence\s+nationale",                       "preference_nationale"),
    (r"immigration\s+clandestine",                          "immigration_clandestine"),
    (r"regroupement\s+familial",                            "regroupement_familial"),
    (r"cartes?\s+de\s+s[ée]jour",                           "carte_sejour"),
    (r"titres?\s+de\s+s[ée]jour",                           "titre_sejour"),
    (r"droit\s+du\s+sol",                                   "droit_sol"),
    (r"droit\s+du\s+sang",                                  "droit_sang"),
    (r"code\s+de\s+la\s+nationalit[ée]",                    "code_nationalite"),
    (r"int[ée]gration\s+r[ée]publicaine",                   "integration_republicaine"),
    (r"identit[ée]\s+nationale",                            "identite_nationale"),

    # ---- Rights and liberties ----
    (r"interruption\s+volontaire\s+de\s+grossesse",         "interruption_volontaire_grossesse"),
    (r"abolition\s+de\s+la\s+peine\s+de\s+mort",            "abolition_peine_mort"),
    (r"libert[ée]\s+de\s+circulation",                      "liberte_circulation"),
    (r"libert[ée]\s+de\s+conscience",                       "liberte_conscience"),
    (r"libert[ée]\s+de\s+la\s+presse",                      "liberte_presse"),
    (r"libert[ée]\s+d['’]?\s*expression",                   "liberte_expression"),
    (r"[ée]galit[ée]\s+des\s+chances",                      "egalite_chances"),
    (r"[ée]galit[ée]\s+hommes?\s+femmes?",                  "egalite_hommes_femmes"),
    (r"justice\s+sociale",                                  "justice_sociale"),
    (r"droits?\s+des\s+femmes?",                            "droits_femmes"),
    (r"droit\s+de\s+vote",                                  "droit_vote"),
    (r"droit\s+au\s+logement",                              "droit_logement"),
    (r"droit\s+[àa]\s+l['’]?\s*[ée]ducation",               "droit_education"),
    (r"droit\s+[àa]\s+la\s+sant[ée]",                       "droit_sante"),
    (r"droit\s+du\s+travail",                               "droit_travail"),
    (r"droit\s+syndical",                                   "droit_syndical"),
    (r"droit\s+de\s+gr[èe]ve",                              "droit_greve"),
    (r"peine\s+de\s+mort",                                  "peine_mort"),

    # ---- School and culture ----
    (r"[ée]coles?\s+publiques?",                            "ecole_publique"),
    (r"[ée]coles?\s+priv[ée]es?",                           "ecole_privee"),
    (r"[ée]coles?\s+la[iï]ques?",                           "ecole_laique"),
    (r"enseignement\s+public",                              "enseignement_public"),
    (r"enseignement\s+priv[ée]",                            "enseignement_prive"),
    (r"guerre\s+scolaire",                                  "guerre_scolaire"),
    (r"[ée]ducation\s+nationale",                           "education_nationale"),

    # ---- Political and institutional life ----
    (r"discours\s+de\s+politique\s+g[ée]n[ée]rale",         "discours_politique_generale"),
    (r"alternance\s+politique",                             "alternance_politique"),
    (r"vote\s+utile",                                       "vote_utile"),
    (r"vote\s+sanction",                                    "vote_sanction"),
    (r"vote\s+de\s+confiance",                              "vote_confiance"),
    (r"motion\s+de\s+censure",                              "motion_censure"),
    (r"question\s+de\s+confiance",                          "question_confiance"),
    (r"politique\s+g[ée]n[ée]rale",                         "politique_generale"),
    (r"conseil\s+restreint",                                "conseil_restreint"),

    # --- Alliances ---

    # 4-party enumerations (anti-establishment discourse: "RPR, PS, UDF, PC")
    (r"\brpr[\s\-/,;]*ps[\s\-/,;]*udf[\s\-/,;]*pc(?:f)?\b",   "rpr_ps_udf_pcf"),
    (r"\brpr[\s\-/,;]*udf[\s\-/,;]*ps[\s\-/,;]*pc(?:f)?\b",   "rpr_ps_udf_pcf"),
    (r"\bps[\s\-/,;]*pc(?:f)?[\s\-/,;]*rpr[\s\-/,;]*udf\b",   "rpr_ps_udf_pcf"),

    # 3-party enumerations
    (r"\brpr[\s\-/,;]*udf[\s\-/,;]*cds\b",                    "rpr_udf_cds"),
    (r"\bps[\s\-/,;]*pc(?:f)?[\s\-/,;]*mrg\b",                "ps_pcf_mrg"),

    # 2-party right coalition (RPR-UDF, all spellings)
    (r"\brpr[\s\-/,;]*udf\b",                                 "rpr_udf"),
    (r"\budf[\s\-/,;]*rpr\b",                                 "udf_rpr"),

    # 2-party left coalitions
    (r"\bps[\s\-/,;]*pc(?:f)?\b",                             "ps_pcf"),
    (r"\bpc(?:f)?[\s\-/,;]*ps\b",                             "pcf_ps"),
]

_COMPOUND_PATTERNS = [
    (re.compile(p, flags=re.IGNORECASE), repl)
    for p, repl in _COMPOUND_PATTERNS_RAW
]

def substitute_compounds(text):
    """Replace multi-word political expressions with single underscore-joined tokens.

    Runs on RAW text, before clean_ocr. This way:
    - We can match apostrophes (`droits de l'homme`) which clean_ocr would strip.
    - The resulting underscore tokens survive spaCy as single PROPN/NOUN units.
    - The lemma equals the substituted form, so no special handling downstream.
    """
    for pat, repl in _COMPOUND_PATTERNS:
        text = pat.sub(repl, text)
    return text

# ============================================================================
# (2-bis) Acronyms — expanded BEFORE everything else, while case is preserved
# ============================================================================
# These run BEFORE clean_ocr because they rely on uppercase to disambiguate.
# Two-letter acronyms (FN, LO) would otherwise be filtered out by min_token_len.

_ACRONYM_PATTERNS_RAW = [
    # ---- Parties (2 letters first since they're the highest risk) ----
    (r"FN",   "front_national"),
    (r"LO",   "lutte_ouvriere"),
    # ---- Parties (3-4 letters) ----
    (r"PCF",  "parti_communiste_francais"),
    (r"PSU",  "parti_socialiste_unifie"),
    (r"RPR",  "rassemblement_pour_republique"),
    (r"UDF",  "union_democratie_francaise"),
    (r"CDS",  "centre_democrates_sociaux"),
    (r"MRG",  "mouvement_radicaux_gauche"),
    (r"LCR",  "ligue_communiste_revolutionnaire"),
    (r"UDR",  "union_democrates_republique"),
    (r"MRP",  "mouvement_republicain_populaire"),
    # ---- Concepts and policies ----
    (r"CEE",  "communaute_economique_europeenne"),
    (r"PAC",  "politique_agricole_commune"),
    (r"RMI",  "revenu_minimum_insertion"),
    (r"IGF",  "impot_grandes_fortunes"),
    (r"ISF",  "impot_fortune"),
    (r"IVG",  "interruption_volontaire_grossesse"),
    (r"SMIC", "salaire_minimum"),
    (r"HLM",  "habitation_loyer_modere"),
    # ---- Unions ----
    (r"CFDT", "cfdt"),   # already a single token, just to keep it lowercase
    (r"CGT",  "cgt"),
    (r"CFTC", "cftc"),
    (r"FEN",  "fen"),
    (r"UNEF", "unef"),
]

# Compile with strict boundaries: only match if surrounded by non-letter chars
# (whitespace, punctuation, start/end of string). Case-SENSITIVE — we rely on
# uppercase form to disambiguate from real lowercase words.
_ACRONYM_PATTERNS = [
    (re.compile(rf"(?<![A-Za-zÀ-ÿ]){acronym}(?![A-Za-zÀ-ÿ])"), repl)
    for acronym, repl in _ACRONYM_PATTERNS_RAW
]


def substitute_acronyms(text):
    """Expand political acronyms BEFORE case is lost.

    Runs first in the pipeline, so it sees the original case (FN, LO, RPR).
    The expansion form is lowercase with underscores, identical to the
    compound form, so downstream stages treat them uniformly.
    """
    for pat, repl in _ACRONYM_PATTERNS:
        text = pat.sub(repl, text)
    return text

# ============================================================================
# (3) OCR cleanup — same as before but allows underscores through
# ============================================================================


_ARCHIVAL_PATTERNS = [
    r"sciences\s*po\s*[/\\\-]*\s*fonds\s*cevipof",
    r"\bcevipof\b",
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
    text = re.sub(r"([a-zàâçéèêëîïôûùüÿñæœ])-\s*([a-zàâçéèêëîïôûùüÿñæœ])",r"\1\2", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ_\s]", " ", text)
    text = re.sub(r"\b[^\W_]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================================================
# (4) Pre-lemma substitutions for first names that corrupt lemmatization
# ============================================================================
# Only used for genuine corruption cases. Plain first names (Jean, Hervé,
# Pierre...) lemmatize correctly to themselves — they go in EXTRA_STOPWORDS,
# not here. The only common case where lemmatization breaks is `Marie` ->
# `marier` (verb).

_PROBLEMATIC_FIRST_NAMES = re.compile(
    r"\b(marie|christian|christiane)\b",  
    re.IGNORECASE,
)


def pre_lemma_substitutions(text):
    """Run AFTER clean_ocr but BEFORE spaCy: scrub first names whose lemma
    collides with a real French verb or adjective."""
    return _PROBLEMATIC_FIRST_NAMES.sub(" ", text)


# ============================================================================
# (5) Stopword list — applied AFTER lemmatization, on lemma forms only
# ============================================================================
# All entries are LEMMAS (singular masculine for nouns/adjectives, infinitive
# for verbs).

EXTRA_STOPWORDS = {
    # --- Electoral vocabulary ---
    # Is it really wise to add it, when we know that there was a topic of alliances between first and second-tour ? 
    # Maybe then the words like "rassemblement", "alliance", "union" will be more present in this topic
    "decouper", "postal", "joindre", "adresse", "tel",
    "feuille", "recevoir", "desirer", "comporter", "complet",
    "reponse", "gratuitement", "panorama", "offset",
    "francais","candidat", "electeur", "electrice", "election", "scrutin", "suffrage",
    "circonscription",  "depute", "voter", "vote", "voix",
    "tour", "dimanche", "suffrage", "sortant", "urne",
    "candidature", "reelection", "bulletin", "canton",
    "monsieur", "madame", "cher", "suppleant", "departement",
    "janvier", "février", "fevrier", "mars", "avril", "mai", "juin",
    "juillet", "août", "aout", "septembre", "octobre", "novembre",
    "décembre", "decembre", "legislatif", "prochain",

    # --- Generic verbs (>40% doc frequency, no thematic content) --- 
    "pouvoir", "devoir", "vouloir", "falloir", "voir", "faire", "donner", 
    "mettre", "permettre",  "agir", "aller", "savoir","choisir","remercier",
    "vivre", "croire", "rendre", "prendre", "trouver", "dire", "venir",
    "tenir", "porter", "passer", "rester", "arriver", "sortir", "partir",
    "entendre", "regarder", "appeler", "rejoindre", "obtenir", "proteger",
    "apporter", "presenter", "assurer", "proposer","defendre", "creer",
    "connaitre","appliquer","compter","continuer", "representer","engager",

    # --- Generic nouns/adjectives ubiquitous to political manifestos ---
    #  "paix", "aide"
    # should I add them ? True that it is not super distinctive
    "politique", "social", "pays", "france", "national", "etat", 
    "vie", "homme", "grand", "bon", "temps", "monde", "jour", 
    "face", "voie", "moyen", "force", "raison", "an", "année", "fin",
    "début", "suite", "réel", "vrai", "fois", "plan", "sens",
    "petit", "nouveau", "ancien", "premier", "dernier", "seul", "tout",
    "même", "autre", "important", "suppléant", "titulaire",
    "programme","gouvernement","parti","majorite","volonte",
    "mesure","besoin", "solution","valeur","interet","ensemble",
    "action", "choix","soutien","soutenir","confiance","priorite",
    "veritable","actuel","creation","groupe","chose","place",
    "tete","nombre","mois","semaine","cote","affaire","responsable",
    "verite","dirigeant","votant",

    # --- German OCR (most paragraphs already removed by strip_german_paragraphs;
    # this catches isolated leftover words that survived in mixed paragraphs) ---
    "die", "der", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
    "und", "oder", "mit", "von", "zu", "zur", "zum", "bei", "für", "fur",
    "ist", "sind", "sie", "wir", "auch", "nicht", "hat", "haben", "wird",
    "werden", "partei", "hlen", "halt", "leben", "arbeiter", "programm",
    "frankreich", "deutsch", "deutschland", "linken",

    # --- First names of candidates that lemmatize cleanly to themselves
    # but flood topics with a single candidate's identity ---
    "jean", "michel", "bruno", "jacques", "pierre", "françois", "francois",
    "paul", "henri", "louis", "claude", "andré", "andre", "philippe",
    "marcel", "robert", "georges", "daniel", "bernard", "alain", "hervé",
    "herve", "brice", 

    # --- OCR garbage that ended up in early topics (toponyms badly
    # lemmatized, fragments that survived) ---
    "vilain", "renne", "vosge", "ille", "ruffi", "masson", "laguiller",
    "arlette", "accé", "minimun", "documenta", "absenc", "eclogie",
    "vif", "socia", "trafi", "fiscer", "redresse", "compléte", "marseill", "saint",
    "tion", "ment", "alsac",

    # --- Encarts biographiques ---
    "naitre", "ne", "marier", "epouse", "epoux",
}


# ============================================================================
# Main pipeline
# ============================================================================

def preprocess_corpus(
    texts,
    spacy_model="fr_core_news_md",
    keep_pos=("NOUN", "PROPN", "ADJ", "VERB"),
    min_token_len=3,
    apply_german_filter=True,
    apply_compounds=True,
):
    """Lemmatize and filter a list of raw texts.
    
    Returns a list of lists of lemmas, ready for topic modeling.
    """
    nlp = spacy.load(spacy_model, disable=["ner", "parser"])
    # just to be sure that no accents remain
    raw_stopwords = set(nlp.Defaults.stop_words) | EXTRA_STOPWORDS
    stopwords_no_accent = {strip_accents(w.lower()) for w in raw_stopwords}

    keep_pos_set = set(keep_pos)
    
    cleaned = []
    for t in texts:
        x = t
        if apply_german_filter:
            x = strip_german_paragraphs(x)
        if apply_compounds:
            x = substitute_compounds(x)
        x = clean_ocr(x)
        x = pre_lemma_substitutions(x)
        cleaned.append(x)
    
    out = []
    for doc in tqdm(nlp.pipe(cleaned, batch_size=64), total=len(cleaned), desc="Lemmatizing"):
        tokens = []
        for tok in doc:
            is_compound = "_" in tok.text

            if is_compound:
                # Composé qu'on a soudé nous-mêmes : on bypass POS et lemma,
                # on prend la forme de surface telle qu'elle a été substituée.
                word = strip_accents(tok.text.lower())
            else:
                if tok.pos_ not in keep_pos_set:
                    continue
                word = strip_accents(tok.lemma_.lower())
        
            if word in stopwords_no_accent:
                continue
            if len(word) < min_token_len:
                continue
            if not word.replace("_", "").isalpha():
                continue
            tokens.append(word)
        out.append(tokens)
    return out

# ============================================================================
# Diagnostics
# ============================================================================

def inspect_documents(df, tokens, n_samples=5, text_col="text", seed=42,
                      meta_cols=("id", "year", "titulaire-soutien"),
                      n_chars=600, n_tokens_preview=50):
    """Print side-by-side raw text and tokens for a few sampled documents."""
    import random
    random.seed(seed)
    sample_indices = random.sample(range(len(df)), n_samples)

    for idx in sample_indices:
        print("=" * 80)
        print(f"Document {idx}")
        for col in meta_cols:
            if col in df.columns:
                print(f"  {col:8s}: {df.iloc[idx][col]}")
        print(f"  n_chars : {len(df.iloc[idx][text_col])}")
        print(f"  n_tokens: {len(tokens[idx])}")

        print(f"\n--- ORIGINAL TEXT (first {n_chars} chars) ---")
        print(df.iloc[idx][text_col][:n_chars])

        print(f"\n--- TOKENS (first {n_tokens_preview}) ---")
        print(tokens[idx][:n_tokens_preview])
        print()


def summarize_token_frequencies(tokens, suspicious_tokens=None,
                                doc_pct_threshold=50.0,
                                top_n=30, save_path=None):
    """Compute a token-frequency table and flag potential leftover stopwords.

    Returns a DataFrame with columns: token, total, n_docs, doc_pct.
    """
    if suspicious_tokens is None:
        suspicious_tokens = ("sciences", "po", "cevipof", "fonds", "république",
                             "législatif", "die", "fur", "und", "tion", "marie")

    n_docs = len(tokens)

    total_count = Counter()
    doc_freq = defaultdict(int)
    for doc in tokens:
        total_count.update(doc)
        for tok in set(doc):
            doc_freq[tok] += 1

    freq_df = pd.DataFrame({
        "token":   list(total_count.keys()),
        "total":   [total_count[t] for t in total_count],
        "n_docs":  [doc_freq[t]    for t in total_count],
        "doc_pct": [100 * doc_freq[t] / n_docs for t in total_count],
    }).sort_values("total", ascending=False).reset_index(drop=True)

    print("=== Suspicious tokens (should ideally be 0) ===")
    for tok in suspicious_tokens:
        n_with = doc_freq.get(tok, 0)
        flag = " ⚠" if n_with > 0 else ""
        print(f"  '{tok:15s}' in {n_with:5d} docs ({100*n_with/n_docs:5.1f}%){flag}")

    print(f"\n=== Top {top_n} tokens by total count ===")
    for _, r in freq_df.head(top_n).iterrows():
        print(f"  {r['token']:30s} total={int(r['total']):>7d}  in {r['doc_pct']:5.1f}% docs")

    high_df = freq_df[freq_df["doc_pct"] > doc_pct_threshold]
    print(f"\n=== {len(high_df)} tokens in >{doc_pct_threshold:.0f}% of documents ===")
    for _, r in high_df.head(top_n).iterrows():
        print(f"  {r['token']:30s} {r['doc_pct']:5.1f}%  ({int(r['n_docs']):>5d} docs)")

    print(f"\nTotal vocabulary size: {len(freq_df):,}")
    n_singletons = (freq_df["total"] == 1).sum()
    print(f"Tokens appearing only once: {n_singletons:,} ({100*n_singletons/len(freq_df):.1f}%)")

    if save_path is not None:
        freq_df.to_csv(save_path, index=False)
        print(f"\nSaved frequency table to {save_path}")

    return freq_df


def summarize_compound_hits(tokens, top_n=40):
    """Show which compound expressions actually appeared in the corpus.

    Run this after preprocessing to validate the regex list. Compounds with
    very few hits can be removed from _COMPOUND_PATTERNS; missing expected
    hits indicate a regex that's not matching.
    """
    counts = Counter()
    for doc in tokens:
        counts.update(t for t in doc if "_" in t)

    print(f"=== Found {len(counts)} distinct compound tokens ===")
    print(f"Total compound occurrences: {sum(counts.values()):,}")
    print(f"\n=== Top {top_n} ===")
    for tok, n in counts.most_common(top_n):
        print(f"  {tok:45s} {n:>6d}")

    return counts