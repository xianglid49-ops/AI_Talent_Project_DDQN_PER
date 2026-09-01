from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    numeric_features: np.ndarray
    skill_matrix: np.ndarray
    skill_columns: List[str]
    text_columns: List[str]
    group_columns: List[str]
    numeric_columns: List[str]
    row_text: List[str]
    role_values: np.ndarray
    course_values: np.ndarray
    demand_signal: np.ndarray
    provenance: Dict


def _normalise_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")


def _find_csv(path: Path, pattern: str = "*.csv") -> Path:
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    candidates = sorted(path.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found under {path}")
    # Prefer the largest CSV because Kaggle downloads sometimes include metadata files.
    return max(candidates, key=lambda p: p.stat().st_size)


def acquire_dataset(
    local_csv: Optional[str],
    download: bool,
    kaggle_slug: str,
    csv_glob: str = "*.csv",
) -> Tuple[Path, Dict]:
    if local_csv:
        csv_path = Path(local_csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        if csv_path.is_dir():
            csv_path = _find_csv(csv_path, csv_glob)
        return csv_path, {
            "source_type": "user_supplied_local_csv",
            "path": str(csv_path),
            "kaggle_slug": kaggle_slug,
        }

    if not download:
        raise ValueError("Supply --data <csv> or use --download.")

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for --download. Install requirements.txt."
        ) from exc

    download_dir = Path(kagglehub.dataset_download(kaggle_slug))
    csv_path = _find_csv(download_dir, csv_glob)
    return csv_path, {
        "source_type": "kagglehub",
        "kaggle_slug": kaggle_slug,
        "download_dir": str(download_dir),
        "path": str(csv_path),
    }


def _matching_columns(
    columns: Sequence[str], candidates: Sequence[str]
) -> List[str]:
    lookup = {_normalise_name(c): c for c in columns}
    found = []
    for candidate in candidates:
        key = _normalise_name(candidate)
        if key in lookup:
            found.append(lookup[key])
    return found


def _infer_skill_columns(
    df: pd.DataFrame,
    numeric_columns: Sequence[str],
    exclude_patterns: Sequence[str],
    min_skill_columns: int,
) -> List[str]:
    excluded = tuple(_normalise_name(p) for p in exclude_patterns)
    skill_cols = []
    for col in numeric_columns:
        norm = _normalise_name(col)
        if any(token in norm for token in excluded):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue
        # Skills are commonly bounded or count-like. Constant administrative
        # identifiers are filtered by requiring variance.
        if float(non_null.std(ddof=0)) <= 1e-12:
            continue
        skill_cols.append(col)

    if len(skill_cols) < min_skill_columns:
        # Fall back to the most variable numerical columns rather than silently failing.
        ranked = []
        for col in numeric_columns:
            s = pd.to_numeric(df[col], errors="coerce")
            ranked.append((float(s.var(skipna=True) or 0.0), col))
        skill_cols = [c for _, c in sorted(ranked, reverse=True)[:max(min_skill_columns, 3)]]
    return skill_cols


def _infer_signal_column(df: pd.DataFrame, terms: Sequence[str]) -> Optional[str]:
    for col in df.columns:
        norm = _normalise_name(col)
        if any(term in norm for term in terms):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def _clean_categorical(series: pd.Series) -> np.ndarray:
    return (
        series.astype("string")
        .fillna("<missing>")
        .str.strip()
        .str.lower()
        .to_numpy(dtype=object)
    )


def load_and_preprocess(
    csv_path: str | Path,
    source_metadata: Dict,
    cfg: Dict,
) -> DatasetBundle:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if cfg.get("max_rows"):
        df = df.head(int(cfg["max_rows"])).copy()

    original_columns = list(df.columns)
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)

    # Convert numeric-looking object columns if most values are numeric.
    for col in df.columns:
        if col in numeric_columns:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.95:
            df[col] = converted
            numeric_columns.append(col)

    text_columns = _matching_columns(
        list(df.columns), cfg.get("text_column_candidates", [])
    )
    group_columns = _matching_columns(
        list(df.columns), cfg.get("group_column_candidates", [])
    )

    skill_columns = _infer_skill_columns(
        df,
        numeric_columns,
        cfg.get("skill_exclude_patterns", []),
        int(cfg.get("min_skill_columns", 3)),
    )

    if not numeric_columns:
        raise ValueError("The dataset contains no usable numeric columns.")

    numeric_imputer = SimpleImputer(strategy="median")
    numeric_raw = numeric_imputer.fit_transform(df[numeric_columns])
    numeric_scaler = MinMaxScaler()
    numeric_features = numeric_scaler.fit_transform(numeric_raw).astype(np.float32)

    skill_indices = [numeric_columns.index(c) for c in skill_columns]
    skill_matrix = numeric_features[:, skill_indices].astype(np.float32)

    # Build a stable textual representation without inventing resume text.
    if text_columns:
        row_text = []
        for _, row in df[text_columns].iterrows():
            parts = []
            for col in text_columns:
                value = row[col]
                if pd.isna(value):
                    continue
                parts.append(f"{_normalise_name(col)}: {str(value).strip()}")
            row_text.append(" | ".join(parts) if parts else "<empty>")
    else:
        row_text = [
            " | ".join(
                f"{_normalise_name(c)}={float(v):.4f}"
                for c, v in zip(skill_columns, skill_matrix[i])
            )
            for i in range(len(df))
        ]

    role_col = next(
        (c for c in group_columns if "role" in _normalise_name(c) or "job" in _normalise_name(c)),
        None,
    )
    course_col = next(
        (c for c in group_columns if "course" in _normalise_name(c)),
        None,
    )

    role_values = (
        _clean_categorical(df[role_col])
        if role_col
        else np.array(["<unknown_role>"] * len(df), dtype=object)
    )
    course_values = (
        _clean_categorical(df[course_col])
        if course_col
        else np.array(["<unknown_course>"] * len(df), dtype=object)
    )

    demand_col = _infer_signal_column(
        df, ["demand", "importance", "market_need", "industry_need"]
    )
    if demand_col is not None:
        vals = pd.to_numeric(df[demand_col], errors="coerce").to_numpy().reshape(-1, 1)
        vals = SimpleImputer(strategy="median").fit_transform(vals)
        demand_signal = MinMaxScaler().fit_transform(vals).ravel().astype(np.float32)
    else:
        # Derive a dataset-only demand proxy from the mean normalized skill intensity.
        demand_signal = skill_matrix.mean(axis=1).astype(np.float32)

    provenance = {
        **source_metadata,
        "rows": int(len(df)),
        "original_columns": original_columns,
        "numeric_columns": numeric_columns,
        "text_columns_used": text_columns,
        "group_columns_used": group_columns,
        "skill_columns_inferred": skill_columns,
        "demand_signal_source": demand_col or "mean_normalized_skill_intensity_proxy",
        "dataset_derived_variables": {
            "skill_vector": skill_columns,
            "text_representation": text_columns,
            "role_or_job_group": role_col,
            "course_group": course_col,
            "demand_signal": demand_col or "derived_from_dataset_skill_columns",
        },
        "simulation_generated_variables": [
            "candidate_id",
            "project_id",
            "candidate_capability_drift",
            "project_requirement_drift",
            "availability",
            "behavioral_compatibility",
            "collaboration_compatibility",
            "experience_relevance",
            "learning_velocity",
            "project_complexity",
            "feedback_signal",
            "reward",
        ],
        "interpretation_warning": (
            "Simulation-generated variables are not observed longitudinal workforce data."
        ),
    }

    return DatasetBundle(
        frame=df,
        numeric_features=numeric_features,
        skill_matrix=skill_matrix,
        skill_columns=skill_columns,
        text_columns=text_columns,
        group_columns=group_columns,
        numeric_columns=numeric_columns,
        row_text=row_text,
        role_values=role_values,
        course_values=course_values,
        demand_signal=demand_signal,
        provenance=provenance,
    )


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.clip(denominator, 1e-8, None)


def make_pair_indices(n_rows: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create one candidate-project pair per source row while keeping candidate
    and project source records distinct whenever possible.
    """
    candidate_idx = np.arange(n_rows, dtype=np.int64)
    project_idx = rng.permutation(n_rows).astype(np.int64)
    same = candidate_idx == project_idx
    if n_rows > 1 and same.any():
        project_idx[same] = np.roll(project_idx, 1)[same]
    return candidate_idx, project_idx


def build_heuristic_labels(
    bundle: DatasetBundle,
    candidate_idx: np.ndarray,
    project_idx: np.ndarray,
    label_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict[str, np.ndarray]]:
    """
    Explicit heuristic supervision.

    Ground truth is based only on dataset-derived variables. Simulated
    availability/behavior/collaboration variables are intentionally excluded.
    """
    c_skill = bundle.skill_matrix[candidate_idx]
    p_skill = bundle.skill_matrix[project_idx]
    skill_similarity = np.clip(cosine_rows(c_skill, p_skill), 0.0, 1.0)

    role_match = (
        bundle.role_values[candidate_idx] == bundle.role_values[project_idx]
    ).astype(np.float32)
    course_match = (
        bundle.course_values[candidate_idx] == bundle.course_values[project_idx]
    ).astype(np.float32)

    # High demand on the project side increases the compatibility contribution
    # of a well-aligned candidate without introducing simulated variables.
    demand_alignment = 1.0 - np.abs(
        bundle.demand_signal[candidate_idx] - bundle.demand_signal[project_idx]
    )
    demand_alignment = np.clip(demand_alignment, 0.0, 1.0).astype(np.float32)

    ws = float(label_cfg["skill_weight"])
    wr = float(label_cfg["role_weight"])
    wc = float(label_cfg["course_weight"])
    wd = float(label_cfg["demand_weight"])
    total = ws + wr + wc + wd
    if total <= 0:
        raise ValueError("Label weights must sum to a positive value.")
    ws, wr, wc, wd = [x / total for x in (ws, wr, wc, wd)]

    compatibility = (
        ws * skill_similarity
        + wr * role_match
        + wc * course_match
        + wd * demand_alignment
    ).astype(np.float32)

    threshold = float(label_cfg["threshold"])
    labels = (compatibility >= threshold).astype(np.int64)

    definition = {
        "formula": (
            "C = w_skill*S_skill + w_role*S_role + "
            "w_course*S_course + w_demand*S_demand; y = 1[C >= threshold]"
        ),
        "normalized_weights": {
            "skill": ws,
            "role": wr,
            "course": wc,
            "demand": wd,
        },
        "threshold": threshold,
        "uses_simulated_variables": False,
        "interpretation": (
            "Heuristic compatibility agreement, not independently observed hiring/project success."
        ),
    }
    components = {
        "skill_similarity": skill_similarity.astype(np.float32),
        "role_match": role_match,
        "course_match": course_match,
        "demand_alignment": demand_alignment,
    }
    return labels, compatibility, definition, components


def write_provenance(bundle: DatasetBundle, out_dir: Path, label_definition: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").write_text(
        json.dumps(bundle.provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "label_definition.json").write_text(
        json.dumps(label_definition, indent=2, ensure_ascii=False), encoding="utf-8"
    )
