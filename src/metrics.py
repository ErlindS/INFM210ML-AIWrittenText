"""Feste Evaluationsmetriken für alle Experimente.

Primär   ROC-AUC  – schwellenwertfrei, invariant gegenüber dem Klassenverhältnis
                    (das je LOGO-Fold variiert), Metrik der Kaggle-Competition.
Sekundär F1 (Klasse KI) – Schwelle wird auf dem *Validierungs*-Split gewählt
                    (max. F1), nie auf Test. Im LOGO-Fold enthält Val den
                    zurückgehaltenen Generator nicht -> realistische, "blinde" Schwelle.
Zusatz   TPR@1%FPR – Anteil erkannter KI-Essays, wenn höchstens 1 % der
                    menschlichen Essays fälschlich markiert werden
                    (Falschbeschuldigung von Studierenden ist der teure Fehler).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve


def choose_threshold(y_val, score_val) -> float:
    """Schwelle, die F1 auf dem Validierungs-Split maximiert."""
    p, r, t = precision_recall_curve(y_val, score_val)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def tpr_at_fpr(y, score, max_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= max_fpr].max())


def evaluate(y, score, threshold: float) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "f1": float(f1_score(y, score >= threshold)),
        "tpr@1%fpr": tpr_at_fpr(y, score, 0.01),
        "threshold": float(threshold),
        "n": int(len(y)),
        "n_ai": int(y.sum()),
    }


def bootstrap_auc_ci(y, score, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """95%-Konfidenzintervall der ROC-AUC (stratifiziertes Bootstrap)."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return tuple(np.quantile(aucs, [alpha / 2, 1 - alpha / 2]))
