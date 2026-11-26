"""Gemeinsames Laden + Bereinigen des DAIGT-v2-Datensatzes.

Alle späteren Skripte (Baseline, Transformer, LOGO) laden die Daten über
`load_essays()`, damit überall exakt dieselbe Datenbasis verwendet wird.

Bereinigungsschritte (in dieser Reihenfolge):
  1. Text-Normalisierung (`normalize_text`): entfernt reine Formatierungs-
     Artefakte der Datensammlung, z. B. führende Leerzeichen
     (99.7 % der Cohere- und 54 % der Mistral-Essays, 0 % der Menschen).
  2. Generator-Familie zuordnen (`GENERATOR`); KI-Essays ohne bekannte Quelle
     (3 Essays aus `train_essays`) werden verworfen.
  3. Degenerierte KI-Texte (< MIN_WORDS Wörter) verwerfen.
  4. Near-Duplicates entfernen (TF-IDF-Kosinus > DUP_THRESHOLD). Wichtig:
     1 294 der Kaggle-`train_essays` sind fast identische Kopien von
     Persuade-Essays -> ohne Dedup landet derselbe Essay in Train und Test.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_CSV = DATA / "daigt_v2" / "train_v2_drcat_02.csv"
DUP_CSV = DATA / "near_duplicates.csv"

# Dokumentation der Datensatz-Version (Kaggle: thedrcat/daigt-v2-train-dataset)
DATASET_SHA256 = "f0c812600ff4f22a11e497816ff3bf88776e570117d355d2c5646cacc65e565c"

SEED = 42
MIN_WORDS = 50
DUP_THRESHOLD = 0.9

# Quelle -> Generator-Familie. LOGO hält eine ganze *Familie* zurück
# (z. B. Llama-2-7B und Llama-2-70B zusammen), sonst wäre "ungesehen" geschönt.
GENERATOR = {
    "chat_gpt_moth": "gpt", "radek_500": "gpt", "radekgpt4": "gpt",
    "llama2_chat": "llama2", "llama_70b_v1": "llama2", "NousResearch/Llama-2-7b-chat-hf": "llama2",
    "mistral7binstruct_v1": "mistral", "mistral7binstruct_v2": "mistral",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
    "darragh_claude_v6": "claude", "darragh_claude_v7": "claude",
    "kingki19_palm": "palm", "palm-text-bison1": "palm",
    "falcon_180b_v1": "falcon",
    "cohere-command": "cohere",
}

_MANY_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    """Konservative Normalisierung: nur Artefakte, keine Stilmerkmale.

    Doppelte Leerzeichen, Tippfehler, Groß-/Kleinschreibung bleiben erhalten –
    das sind echte (menschliche) Schreibmerkmale.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def essay_id(text: str) -> str:
    """Stabile ID aus dem normalisierten Text (unabhängig von Zeilenreihenfolge)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def find_near_duplicates(texts: pd.Series, threshold: float = DUP_THRESHOLD,
                         chunk: int = 1500) -> pd.DataFrame:
    """Alle Paare (i < j) mit TF-IDF-Kosinus > threshold (Wort-1/2-Gramme)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        dtype=np.float32).fit_transform(texts)
    XT = X.T.tocsr()
    pairs = []
    for s in range(0, X.shape[0], chunk):
        S = (X[s:s + chunk] @ XT).toarray()
        rows, cols = np.where(S > threshold)
        keep = s + rows < cols
        pairs.append(np.c_[s + rows[keep], cols[keep], S[rows[keep], cols[keep]]])
    p = np.vstack(pairs) if pairs else np.empty((0, 3))
    return pd.DataFrame({"i": p[:, 0].astype(int), "j": p[:, 1].astype(int), "sim": p[:, 2]})


def _dedup_mask(df: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Union-Find über Duplikat-Paare. Pro Gruppe bleibt ein Essay
    (bevorzugt `persuade_corpus`, weil vollständige Metadaten); Gruppen mit
    widersprüchlichen Labels (Mensch vs. KI) werden komplett verworfen."""
    pos = {eid: k for k, eid in enumerate(df["essay_id"])}
    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["id_a"], pairs["id_b"]):
        if a in pos and b in pos:
            ra, rb = find(pos[a]), find(pos[b])
            if ra != rb:
                parent[rb] = ra
    root = np.array([find(k) for k in range(len(df))])
    keep = np.ones(len(df), dtype=bool)
    g = df.assign(_root=root, _prio=(df["source"] != "persuade_corpus").astype(int),
                  _pos=np.arange(len(df)))
    for _, grp in g[g.duplicated("_root", keep=False)].groupby("_root"):
        keep[grp["_pos"]] = False
        if grp["label"].nunique() == 1:
            keep[grp.sort_values(["_prio", "_pos"])["_pos"].iloc[0]] = True
    return keep


def load_essays(dedup: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Bereinigter Datensatz mit Spalten
    essay_id, text, label, generator, source, prompt_name, n_words, n_chars."""
    raw = load_raw()
    df = raw.copy()
    df["text_raw"] = df["text"]
    df["text"] = df["text"].map(normalize_text)
    df["generator"] = df["source"].map(GENERATOR).where(df["label"] == 1, "human")
    df["n_words"] = df["text"].str.split().str.len()
    df["n_chars"] = df["text"].str.len()
    df["essay_id"] = df["text"].map(essay_id)
    log = {"raw": len(df)}

    df = df.dropna(subset=["generator"])
    log["unknown_generator"] = log["raw"] - len(df)
    n = len(df)
    df = df[~((df["label"] == 1) & (df["n_words"] < MIN_WORDS))]
    log["ai_below_min_words"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates("essay_id")
    log["exact_duplicates"] = n - len(df)
    df = df.reset_index(drop=True)

    if dedup:
        if not DUP_CSV.exists():
            p = find_near_duplicates(df["text"])
            pd.DataFrame({"id_a": df["essay_id"].values[p.i], "id_b": df["essay_id"].values[p.j],
                          "src_a": df["source"].values[p.i], "src_b": df["source"].values[p.j],
                          "sim": p.sim.round(4)}).to_csv(DUP_CSV, index=False)
        pairs = pd.read_csv(DUP_CSV)
        n = len(df)
        df = df[_dedup_mask(df, pairs)].reset_index(drop=True)
        log["near_duplicates"] = n - len(df)
    log["final"] = len(df)
    if verbose:
        print("Bereinigung:", log)
    df.attrs["cleaning_log"] = log
    cols = ["essay_id", "text", "text_raw", "label", "generator", "source", "prompt_name",
            "n_words", "n_chars"]
    return df[cols]


def load_splits() -> pd.DataFrame:
    """Essays + Split-Spalten (random, logo_<gen>) aus `src/splits.py`."""
    splits = pd.read_csv(DATA / "splits.csv")
    return load_essays().merge(splits.drop(columns=["label", "generator", "prompt_name"]),
                               on="essay_id", validate="one_to_one")


def get_fold(df: pd.DataFrame, split_col: str):
    """(train, val, test) für eine Split-Spalte, z. B. 'random' oder 'logo_gpt'."""
    return tuple(df[df[split_col] == part] for part in ("train", "val", "test"))
