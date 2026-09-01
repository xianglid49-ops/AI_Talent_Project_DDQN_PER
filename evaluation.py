from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity


def minmax_score(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def choose_threshold(y_true: np.ndarray, score: np.ndarray, metric: str = "f1") -> float:
    best_t, best_v = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (score >= t).astype(int)
        if metric == "accuracy":
            value = accuracy_score(y_true, pred)
        else:
            value = f1_score(y_true, pred, zero_division=0)
        if value > best_v:
            best_v, best_t = float(value), float(t)
    return best_t


def binary_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (np.asarray(score) >= threshold).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, score))
    except Exception:
        result["roc_auc"] = float("nan")
    return result


def rule_based_score(components: Dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    # Intentionally simpler than the label generator.
    return (
        0.80 * components["skill_similarity"][indices]
        + 0.20 * components["role_match"][indices]
    ).astype(np.float32)


def tfidf_pair_score(
    texts: Sequence[str],
    candidate_idx: np.ndarray,
    project_idx: np.ndarray,
    fit_indices: np.ndarray,
    eval_indices: np.ndarray,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    fit_rows = np.unique(
        np.concatenate([candidate_idx[fit_indices], project_idx[fit_indices]])
    )
    vectorizer.fit([texts[i] for i in fit_rows])

    c = vectorizer.transform([texts[i] for i in candidate_idx[eval_indices]])
    p = vectorizer.transform([texts[i] for i in project_idx[eval_indices]])
    numer = np.asarray(c.multiply(p).sum(axis=1)).ravel()
    denom = np.sqrt(np.asarray(c.multiply(c).sum(axis=1)).ravel()) * np.sqrt(
        np.asarray(p.multiply(p).sum(axis=1)).ravel()
    )
    return (numer / np.clip(denom, 1e-12, None)).astype(np.float32)


def transformer_cosine_score(
    text_embeddings: np.ndarray,
    candidate_idx: np.ndarray,
    project_idx: np.ndarray,
    eval_indices: np.ndarray,
) -> np.ndarray:
    c = text_embeddings[candidate_idx[eval_indices]]
    p = text_embeddings[project_idx[eval_indices]]
    numer = np.sum(c * p, axis=1)
    denom = np.linalg.norm(c, axis=1) * np.linalg.norm(p, axis=1)
    return (numer / np.clip(denom, 1e-8, None)).astype(np.float32)


def topk_hit_rate(
    y_true: np.ndarray,
    score: np.ndarray,
    project_ids: np.ndarray,
    k: int,
) -> float:
    """
    For every synthetic project id, a hit occurs if at least one heuristic-
    compatible candidate is present in the top-k scored pairs for that project.
    Projects with no positive pair are excluded.
    """
    df = pd.DataFrame(
        {"project": project_ids, "y": y_true.astype(int), "score": score}
    )
    hits = []
    for _, group in df.groupby("project"):
        if group["y"].sum() == 0:
            continue
        top = group.nlargest(min(k, len(group)), "score")
        hits.append(float(top["y"].max() == 1))
    return float(np.mean(hits)) if hits else float("nan")


def summarize_metrics(fold_metrics: pd.DataFrame, confidence_level: float = 0.95) -> pd.DataFrame:
    alpha = 1.0 - confidence_level
    rows = []
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for model, group in fold_metrics.groupby("model"):
        for metric in metric_cols:
            vals = group[metric].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            mean = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if len(vals) > 1:
                sem = stats.sem(vals)
                tcrit = stats.t.ppf(1 - alpha / 2, df=len(vals) - 1)
                half = float(tcrit * sem)
            else:
                half = 0.0
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_runs": len(vals),
                    "mean": mean,
                    "std": sd,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                }
            )
    return pd.DataFrame(rows)


def paired_tests(
    fold_metrics: pd.DataFrame,
    proposed_name: str = "Proposed_GNN_Transformer_DDQN_PER",
    metric: str = "f1",
) -> pd.DataFrame:
    base = fold_metrics[fold_metrics["model"] == proposed_name][
        ["seed", "fold", metric]
    ].rename(columns={metric: "proposed"})
    rows = []
    for model in sorted(set(fold_metrics["model"]) - {proposed_name}):
        other = fold_metrics[fold_metrics["model"] == model][
            ["seed", "fold", metric]
        ].rename(columns={metric: "baseline"})
        merged = base.merge(other, on=["seed", "fold"], how="inner")
        if len(merged) < 2:
            continue
        stat, p = stats.ttest_rel(
            merged["proposed"].to_numpy(), merged["baseline"].to_numpy()
        )
        diff = merged["proposed"] - merged["baseline"]
        rows.append(
            {
                "baseline": model,
                "metric": metric,
                "n_pairs": len(merged),
                "mean_difference": float(diff.mean()),
                "t_statistic": float(stat),
                "p_value": float(p),
            }
        )
    return pd.DataFrame(rows)


def run_shap_on_prediction_head(
    prediction_head,
    latent: np.ndarray,
    scalar: np.ndarray,
    scalar_names: Sequence[str],
    validation_indices: np.ndarray,
    explain_indices: np.ndarray,
    out_path: Path,
    background_size: int = 500,
    explain_size: int = 500,
    device=None,
):
    """
    SHAP explains the final prediction head only.

    Graph/Transformer latent vectors are frozen inputs. This deliberately does
    not claim attribution inside graph message-passing operations.
    """
    import shap
    import torch

    combined = np.concatenate([latent, scalar], axis=1).astype(np.float32)
    bg_idx = validation_indices[: min(background_size, len(validation_indices))]
    ex_idx = explain_indices[: min(explain_size, len(explain_indices))]

    background = torch.as_tensor(combined[bg_idx], dtype=torch.float32, device=device)
    explain_x = torch.as_tensor(combined[ex_idx], dtype=torch.float32, device=device)

    class Wrapper(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, x):
            return self.head.forward_combined(x).unsqueeze(1)

    wrapper = Wrapper(prediction_head).to(device).eval()
    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(explain_x)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    latent_dim = latent.shape[1]
    names = [f"latent_{i:03d}" for i in range(latent_dim)] + list(scalar_names)
    mean_abs = np.abs(shap_values).mean(axis=0)

    detailed = pd.DataFrame(
        {"feature": names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)

    # Aggregate all latent dimensions into one transparent category, while
    # keeping the explicit simulation/compatibility scalars separate.
    aggregate_rows = [
        {
            "feature_group": "graph_transformer_latent",
            "mean_abs_shap": float(mean_abs[:latent_dim].sum()),
        }
    ]
    for i, name in enumerate(scalar_names, start=latent_dim):
        aggregate_rows.append(
            {"feature_group": name, "mean_abs_shap": float(mean_abs[i])}
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        "mean_abs_shap", ascending=False
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out_path.with_name("shap_feature_importance_detailed.csv"), index=False)
    aggregate.to_csv(out_path, index=False)
    return aggregate
