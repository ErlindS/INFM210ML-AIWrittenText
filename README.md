# INFM210ML – Can we detect AI-written text?

AI Lab WS 26/27 · Gruppe **GlEr** (Gleb Janickij, Erlind S.)

**Forschungsfrage:** Erkennt ein Detektor für KI-generierte Essays auch Texte eines
Sprachmodells, das er im Training nie gesehen hat? Dazu vergleichen wir den Random-Split
mit Leave-One-Generator-Out (LOGO) auf identischen Testzeilen.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python src/data/KaggleDataDownloader.py      # DAIGT v2 -> src/data/daigt_v2/
```

Datensatz: Kaggle `thedrcat/daigt-v2-train-dataset`, Datei `train_v2_drcat_02.csv`
(SHA-256 `f0c81260…65e565c`, 44 868 Essays).

## Pipeline (Week 1)

```bash
cd src
python splits.py      # Bereinigung + Random-Split + LOGO-Folds -> data/splits.csv
```

Danach `notebooks/week1_data_analysis.ipynb` ausführen: Statistiken -> `results/week1/`,
Abbildungen für die Folien -> `docs/slides/week1/figures/`.

| Datei | Inhalt |
|---|---|
| `src/data_utils.py` | Laden, Text-Normalisierung, Generator-Familien, Near-Duplicate-Entfernung, `load_splits()` / `get_fold()` |
| `src/splits.py` | Random 70/10/20 (stratifiziert nach Generator) und 6 LOGO-Folds |
| `src/metrics.py` | ROC-AUC (primär), F1 mit Val-Schwelle, TPR@1 %FPR, Bootstrap-KI |
| `src/data/splits.csv` | Split-Zuordnung pro `essay_id` (Hash des normalisierten Texts) |
| `src/data/near_duplicates.csv` | Gefundene Near-Duplicate-Paare (Cache, wird bei Bedarf neu berechnet, ~3 min) |
| `notebooks/week1_data_analysis.ipynb` | Week-1-Analyse: Zählungen, Längen, Shortcut-Checks, Plots; Export ohne Ausgaben unter `notebooks/py/` |
| `docs/slides/week1/week1.pdf` | Folien für den Check-in am 27.11. |

In späteren Experimenten werden die Daten so geladen:

```python
from data_utils import load_splits, get_fold
df = load_splits()
train, val, test = get_fold(df, "logo_gpt")   # oder "random"
```
