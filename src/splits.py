"""Random-Split und Leave-One-Generator-Out-(LOGO-)Folds.

Design (gepaarter Vergleich):
  random       70/10/20 train/val/test, stratifiziert nach Generator (inkl. human).
  logo_<g>     train/val = random-train/val OHNE Generator g
               test      = random-test, eingeschränkt auf {human, g}
               alle übrigen Zeilen = "unused"

Damit werten Random-Split und LOGO-Fold g auf *exakt denselben Testzeilen* aus
(menschliche Test-Essays + Test-Essays von g). Der einzige Unterschied: Hat das
Modell g im Training gesehen? -> Die AUC-Differenz ist direkt der
Generalisierungsverlust auf einen ungesehenen Generator.

Ausgabe: src/data/splits.csv (nur IDs + Split-Spalten, kein Text).
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import DATA, SEED, load_essays

MIN_ESSAYS = 1000          # Generatoren mit weniger Essays bekommen keinen eigenen Fold
TEST_SIZE, VAL_SIZE = 0.20, 0.10
RESULTS = Path(__file__).resolve().parent.parent / "results" / "week1"


def make_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    # 1. Generatoren mit genug Essays für einen eigenen LOGO-Fold
    counts = df.loc[df["label"] == 1, "generator"].value_counts()
    selected = sorted(counts[counts >= MIN_ESSAYS].index)

    # 2. Random-Split 70/10/20, stratifiziert nach Generator
    rest, test = train_test_split(df.index, test_size=TEST_SIZE,
                                  stratify=df["generator"], random_state=SEED)
    train, val = train_test_split(rest, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                  stratify=df.loc[rest, "generator"], random_state=SEED)
    df["random"] = "train"
    df.loc[val, "random"] = "val"
    df.loc[test, "random"] = "test"

    # 3. LOGO-Folds
    for g in selected:
        col = f"logo_{g}"
        is_g = df["generator"] == g
        df[col] = "unused"
        trainval = df["random"].isin(["train", "val"]) & ~is_g
        df.loc[trainval, col] = df.loc[trainval, "random"]
        test_rows = (df["random"] == "test") & (is_g | (df["generator"] == "human"))
        df.loc[test_rows, col] = "test"
    return df, selected


def summarize(df: pd.DataFrame, split_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in split_cols:
        for part in ("train", "val", "test"):
            sub = df[df[c] == part]
            rows.append({"split": c, "part": part, "human": int((sub.label == 0).sum()),
                         "ai": int((sub.label == 1).sum()), "total": len(sub)})
    return pd.DataFrame(rows)


def check(df: pd.DataFrame, selected: list[str]) -> None:
    """Sanity-Checks: keine Überlappung, kein g im LOGO-Training, gleiche Testzeilen."""
    for g in selected:
        col = f"logo_{g}"
        assert not ((df[col].isin(["train", "val"])) & (df["generator"] == g)).any()
        logo_test = set(df.index[df[col] == "test"])
        rand_test = set(df.index[(df["random"] == "test") & df["generator"].isin(["human", g])])
        assert logo_test == rand_test, g
        assert (df.loc[df[col] == "test", "generator"].isin(["human", g])).all()
    assert df["essay_id"].is_unique


if __name__ == "__main__":
    df = load_essays(verbose=True)
    df, selected = make_splits(df)
    check(df, selected)
    split_cols = ["random"] + [f"logo_{g}" for g in selected]

    counts = df.groupby("generator").size().sort_values(ascending=False)
    print("Essays pro Generator (nach Bereinigung):\n", counts.to_string(), sep="")
    print(f"\nLOGO-Folds (>= {MIN_ESSAYS} Essays): {selected}")
    summary = summarize(df, split_cols)
    print("\nEssays pro Split:\n",
          summary.pivot(index="split", columns="part", values=["human", "ai"]).to_string(), sep="")

    out = DATA / "splits.csv"
    df[["essay_id", "label", "generator", "prompt_name"] + split_cols].to_csv(out, index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS / "split_summary.csv", index=False)
    print(f"\nGespeichert: {out}")
