"""Loading of the Archelec corpus.

The corpus consists of two heterogeneous sources to merge:

1. **Metadata** — a CSV exported from https://archelec.sciencespo.fr/explorer
   containing one row per political manifesto with fields like
   ``titulaire-nom``, ``titulaire-soutien`` (party), ``titulaire-profession``,
   ``titulaire-sexe``, ``election-date``, ``departement-insee``, etc.

2. **Transcriptions** — OCR'd text files in a Gitlab repository at
   https://gitlab.teklia.com/ckermorvant/arkindex_archelec, organised
   typically by election year and document ID.

Both sources are joined on the document/Arkindex ID to produce a single
pandas.DataFrame ready for downstream analysis.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

from pathlib import Path
import pandas as pd


def load_transcriptions(text_files_dir: str | Path) -> pd.DataFrame:
    """Loads all .txt files recursively from text_files_dir.
    
    Returns a dictionary {id: text} where the id is the filename
    without extension (corresponding to the 'id' column in the CSV).
    """
    text_files_dir = Path(text_files_dir)
    rows = []
    
    # Recursively find all .txt files and read their content
    for txt_path in Path(text_files_dir).rglob("*.txt"):
        rows.append({
            "id": txt_path.stem,    # filename without extension
            "text": txt_path.read_text(encoding="utf-8", errors="replace"),
        })
    return pd.DataFrame(rows)


def load_corpus(metadata_csv: str | Path, text_files_dir: str | Path) -> pd.DataFrame:
    """Loads and joins metadata + transcriptions.
    
    Returns a DataFrame with the id and text column from the "professions de foi"
    and all the metadata columns from the CSV. 
    """
    metadata = pd.read_csv(metadata_csv)
    transcriptions = load_transcriptions(text_files_dir)
    
    # Join : for each row in the CSV, we look up the corresponding text

    df = transcriptions.merge(metadata, on="id", how="left")
    
    return df
