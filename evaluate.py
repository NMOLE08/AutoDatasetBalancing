from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ScoreBreakdown:
    """Composite metric output for optimization and debugging."""

    total_score: float
    performance_score: float
    intrinsic_score: float
    distance_score: float
    macro_f1: float
    mcc: float
    imbalance_score: float
    retention_score: float
    wasserstein_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def _cap_categorical_cardinality(x: pd.DataFrame, max_categories_per_column: int = 30) -> pd.DataFrame:
    capped = x.copy()
    for col in capped.columns:
        if pd.api.types.is_numeric_dtype(capped[col]):
            continue

        non_null = capped[col].dropna()
        if non_null.empty:
            continue

        top_values = non_null.astype(str).value_counts().head(max_categories_per_column).index
        capped[col] = capped[col].astype(str).where(capped[col].astype(str).isin(top_values), "__OTHER__")

    return capped


def _prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    y = df[target_column]
    x = df.drop(columns=[target_column])
    x = _cap_categorical_cardinality(x, max_categories_per_column=30)

    # One-hot encoding gives the fixed evaluator a stable numeric matrix across mixed dtypes.
    x = pd.get_dummies(x, drop_first=False)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0)

    # Hard cap for speed and memory stability in small/medium test loops.
    if x.shape[1] > 600:
        variances = x.var(axis=0)
        keep_cols = variances.sort_values(ascending=False).head(600).index
        x = x.loc[:, keep_cols]

    if x.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing.")

    return x, y


def _compute_performance_score(df: pd.DataFrame, target_column: str, random_state: int) -> tuple[float, float, float]:
    x, y = _prepare_features(df, target_column)

    if len(x) > 4000:
        sampled_idx = y.sample(n=4000, random_state=random_state).index
        x = x.loc[sampled_idx]
        y = y.loc[sampled_idx]

    if y.nunique(dropna=False) < 2:
        return 0.0, 0.0, 0.0

    class_counts = y.value_counts(dropna=False)
    stratify_target = y if int(class_counts.min()) >= 2 else None

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError:
        # Fallback for extremely sparse label spaces.
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=None,
        )

    if y_train.nunique(dropna=False) < 2 or y_test.nunique(dropna=False) < 2:
        return 0.0, 0.0, 0.0

    model = HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=60,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    macro_f1 = _safe_clip(f1_score(y_test, y_pred, average="macro", zero_division=0))
    mcc_raw = matthews_corrcoef(y_test, y_pred)
    mcc = _safe_clip((mcc_raw + 1.0) / 2.0)

    # Performance = 40% Macro-F1 + 20% MCC, then normalized into [0, 1].
    performance = _safe_clip(((0.40 * macro_f1) + (0.20 * mcc)) / 0.60)
    return performance, macro_f1, mcc


def _compute_imbalance_score(df: pd.DataFrame, target_column: str) -> float:
    counts = df[target_column].value_counts(dropna=False)
    if counts.empty:
        return 0.0

    ratio = float(counts.min() / max(counts.max(), 1))
    return _safe_clip(ratio)


def _compute_retention_score(mutated_df: pd.DataFrame, reference_df: pd.DataFrame) -> float:
    if reference_df.empty:
        return 0.0

    row_retention = min(len(mutated_df), len(reference_df)) / max(len(reference_df), 1)

    total_cells = max(mutated_df.shape[0] * max(mutated_df.shape[1], 1), 1)
    nan_ratio = float(mutated_df.isna().sum().sum()) / total_cells
    nan_score = _safe_clip(1.0 - nan_ratio)

    return _safe_clip((0.7 * row_retention) + (0.3 * nan_score))


def _wasserstein_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
    """Computes 1D Wasserstein distance without requiring scipy."""

    u = np.sort(u_values.astype(float))
    v = np.sort(v_values.astype(float))

    if u.size == 0 or v.size == 0:
        return 1.0

    all_values = np.concatenate([u, v])
    all_values.sort()
    deltas = np.diff(all_values)

    if deltas.size == 0:
        return 0.0

    u_cdf = np.searchsorted(u, all_values[:-1], side="right") / u.size
    v_cdf = np.searchsorted(v, all_values[:-1], side="right") / v.size

    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def _compute_distance_similarity(mutated_df: pd.DataFrame, reference_df: pd.DataFrame, target_column: str) -> float:
    """Compares minority class feature distributions between reference and mutated datasets."""

    if target_column not in mutated_df.columns or target_column not in reference_df.columns:
        return 0.0

    ref_counts = reference_df[target_column].value_counts(dropna=False)
    if ref_counts.empty:
        return 0.0

    minority_class = ref_counts.idxmin()

    ref_min = reference_df[reference_df[target_column] == minority_class].copy()
    mut_min = mutated_df[mutated_df[target_column] == minority_class].copy()

    if ref_min.empty or mut_min.empty:
        return 0.0

    if len(ref_min) > 1000:
        ref_min = ref_min.sample(n=1000, random_state=42)
    if len(mut_min) > 1000:
        mut_min = mut_min.sample(n=1000, random_state=42)

    ref_x = pd.get_dummies(ref_min.drop(columns=[target_column]), drop_first=False)
    mut_x = pd.get_dummies(mut_min.drop(columns=[target_column]), drop_first=False)

    aligned_ref, aligned_mut = ref_x.align(mut_x, axis=1, fill_value=0)

    numeric_cols = aligned_ref.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 0.5

    if len(numeric_cols) > 64:
        numeric_cols = numeric_cols[:64]

    distances: list[float] = []
    for col in numeric_cols:
        u = aligned_ref[col].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        v = aligned_mut[col].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
        distances.append(_wasserstein_1d(u, v))

    if not distances:
        return 0.5

    avg_distance = float(np.mean(distances))
    # Larger distances indicate low-fidelity synthetic data; invert to similarity.
    return _safe_clip(1.0 / (1.0 + avg_distance))


def evaluate_dataset(
    mutated_df: pd.DataFrame,
    target_column: str,
    reference_df: pd.DataFrame,
    random_state: int = 42,
) -> ScoreBreakdown:
    """
    Calculates the full Dataset Health Score.

    Total_Score = 0.60 * Performance + 0.25 * Intrinsic + 0.15 * Distance
    """

    if mutated_df.empty:
        raise ValueError("Mutated dataset is empty.")

    if target_column not in mutated_df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the mutated dataset.")

    performance, macro_f1, mcc = _compute_performance_score(mutated_df, target_column, random_state)

    imbalance_score = _compute_imbalance_score(mutated_df, target_column)
    retention_score = _compute_retention_score(mutated_df, reference_df)

    # Intrinsic = 15% imbalance + 10% retention, normalized into [0, 1].
    intrinsic = _safe_clip(((0.15 * imbalance_score) + (0.10 * retention_score)) / 0.25)

    wasserstein_similarity = _compute_distance_similarity(mutated_df, reference_df, target_column)
    distance = _safe_clip(wasserstein_similarity)

    total = _safe_clip((0.60 * performance) + (0.25 * intrinsic) + (0.15 * distance))

    return ScoreBreakdown(
        total_score=total,
        performance_score=performance,
        intrinsic_score=intrinsic,
        distance_score=distance,
        macro_f1=macro_f1,
        mcc=mcc,
        imbalance_score=imbalance_score,
        retention_score=retention_score,
        wasserstein_similarity=wasserstein_similarity,
    )
