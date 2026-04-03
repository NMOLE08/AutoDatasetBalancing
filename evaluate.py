from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split


MISSING_TOKENS = {"?", "NA", "N/A", "none", "null", ""}


@dataclass(frozen=True)
class ScoreBreakdown:
    total_score: float
    performance_score: float
    logic_score: float
    intrinsic_score: float
    distance_score: float

    macro_f1: float
    mcc: float

    type_integrity_score: float
    bound_plausibility_score: float
    nan_resolution_score: float

    imbalance_score: float
    retention_score: float
    wasserstein_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            out[col] = out[col].apply(
                lambda v: np.nan
                if (isinstance(v, str) and v.strip().lower() in MISSING_TOKENS)
                else v
            )
    return out


def check_domain_logic(
    df: pd.DataFrame,
    domain_rules: dict[str, dict[str, Any]],
    target_column: str,
) -> tuple[float, float, float, float]:
    """
    Returns: logic_score, type_integrity_score, bound_plausibility_score, nan_resolution_score
    """

    # Severe edge-case penalties requested for unresolved missing markers.
    extra_logic_penalty = 0.0
    contains_question_mark = bool(
        df.apply(
            lambda col: col.astype(str).eq("?").any()
            if (pd.api.types.is_string_dtype(col) or col.dtype == object)
            else False
        ).any()
    )
    if contains_question_mark:
        extra_logic_penalty += 0.20

    contains_unresolved_nan = bool(df.isna().any().any())
    if contains_unresolved_nan:
        extra_logic_penalty += 0.10

    work = _normalize_missing(df)

    checked_values = 0
    type_violations = 0
    bound_violations = 0

    for col, rule in domain_rules.items():
        if col not in work.columns:
            continue

        series = work[col]
        numeric = pd.to_numeric(series, errors="coerce")
        non_missing_mask = ~series.isna()
        non_missing_count = int(non_missing_mask.sum())
        checked_values += non_missing_count

        if non_missing_count == 0:
            continue

        if rule.get("type") == "int":
            valid_type = non_missing_mask & numeric.notna() & np.isfinite(numeric)
            valid_type &= np.isclose(numeric, np.round(numeric), atol=1e-9)
            type_violations += int((non_missing_mask & ~valid_type).sum())

        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None or max_v is not None:
            valid_bound = non_missing_mask & numeric.notna() & np.isfinite(numeric)
            if min_v is not None:
                valid_bound &= numeric >= float(min_v)
            if max_v is not None:
                valid_bound &= numeric <= float(max_v)
            bound_violations += int((non_missing_mask & ~valid_bound).sum())

    if checked_values == 0:
        type_integrity_score = 0.0
        bound_plausibility_score = 0.0
    else:
        type_integrity_score = _safe_clip(1.0 - (type_violations / checked_values))
        bound_plausibility_score = _safe_clip(1.0 - (bound_violations / checked_values))

    feature_cols = [c for c in work.columns if c != target_column]
    if not feature_cols:
        nan_resolution_score = 0.0
    else:
        unresolved = int(work[feature_cols].isna().sum().sum())
        total_cells = int(work[feature_cols].shape[0] * work[feature_cols].shape[1])
        nan_resolution_score = _safe_clip(1.0 - (unresolved / max(total_cells, 1)))

    # Inside Logic (25% total): Type 10%, Bounds 10%, NaN 5% => 0.4, 0.4, 0.2
    logic_score = _safe_clip(
        (0.4 * type_integrity_score)
        + (0.4 * bound_plausibility_score)
        + (0.2 * nan_resolution_score)
    )
    logic_score = _safe_clip(logic_score - extra_logic_penalty)

    return logic_score, type_integrity_score, bound_plausibility_score, nan_resolution_score


def _prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    work = _normalize_missing(df)
    y = work[target_column].copy()
    x = work.drop(columns=[target_column])

    x = pd.get_dummies(x, drop_first=False)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0.0)

    if x.shape[1] == 0:
        raise ValueError("No usable feature columns after preprocessing.")

    if y.dtype == object:
        y = y.astype(str)

    return x, y


def _compute_performance_score(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> tuple[float, float, float]:
    x, y = _prepare_features(df, target_column)

    if y.nunique(dropna=False) < 2:
        return 0.0, 0.0, 0.0

    counts = y.value_counts(dropna=False)
    stratify_target = y if int(counts.min()) >= 2 else None

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=stratify_target,
        )
    except ValueError:
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
        max_iter=80,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    macro_f1 = _safe_clip(f1_score(y_test, y_pred, average="macro", zero_division=0))
    mcc_raw = matthews_corrcoef(y_test, y_pred)
    mcc = _safe_clip((mcc_raw + 1.0) / 2.0)

    # Inside Performance (45% total): Macro F1 30%, MCC 15% => 2:1 ratio.
    performance = _safe_clip((2.0 * macro_f1 + mcc) / 3.0)
    return performance, macro_f1, mcc


def _compute_imbalance_score(df: pd.DataFrame, target_column: str) -> float:
    if target_column not in df.columns:
        return 0.0
    counts = df[target_column].value_counts(dropna=False)
    if counts.empty:
        return 0.0
    return _safe_clip(float(counts.min()) / max(float(counts.max()), 1.0))


def _compute_retention_score(mutated_df: pd.DataFrame, reference_df: pd.DataFrame) -> float:
    if reference_df.empty:
        return 0.0
    return _safe_clip(len(mutated_df) / max(len(reference_df), 1))


def _wasserstein_1d(u_values: np.ndarray, v_values: np.ndarray) -> float:
    u = np.sort(u_values.astype(float))
    v = np.sort(v_values.astype(float))
    if u.size == 0 or v.size == 0:
        return 1.0

    all_values = np.sort(np.concatenate([u, v]))
    deltas = np.diff(all_values)
    if deltas.size == 0:
        return 0.0

    u_cdf = np.searchsorted(u, all_values[:-1], side="right") / u.size
    v_cdf = np.searchsorted(v, all_values[:-1], side="right") / v.size
    return float(np.sum(np.abs(u_cdf - v_cdf) * deltas))


def _compute_distance_similarity(
    mutated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    target_column: str,
) -> float:
    if mutated_df.empty or reference_df.empty:
        return 0.0

    mut = _normalize_missing(mutated_df)
    ref = _normalize_missing(reference_df)

    mut_x = mut.drop(columns=[target_column], errors="ignore")
    ref_x = ref.drop(columns=[target_column], errors="ignore")

    mut_x = pd.get_dummies(mut_x, drop_first=False)
    ref_x = pd.get_dummies(ref_x, drop_first=False)
    ref_x, mut_x = ref_x.align(mut_x, axis=1, fill_value=0)

    numeric_cols = ref_x.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 0.5

    if len(numeric_cols) > 128:
        numeric_cols = numeric_cols[:128]

    distances: list[float] = []
    for col in numeric_cols:
        u = pd.to_numeric(ref_x[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        v = pd.to_numeric(mut_x[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        distances.append(_wasserstein_1d(u, v))

    if not distances:
        return 0.5

    avg_distance = float(np.mean(distances))
    similarity = 1.0 / (1.0 + avg_distance)
    return _safe_clip(similarity)


def evaluate_dataset(
    mutated_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    target_column: str,
    domain_rules: dict[str, dict[str, Any]],
    random_state: int = 42,
) -> ScoreBreakdown:
    perf, macro_f1, mcc = _compute_performance_score(mutated_df, target_column, random_state)
    logic, type_score, bound_score, nan_score = check_domain_logic(
        mutated_df,
        domain_rules,
        target_column,
    )

    imbalance = _compute_imbalance_score(mutated_df, target_column)
    retention = _compute_retention_score(mutated_df, reference_df)
    intrinsic = _safe_clip((0.75 * imbalance) + (0.25 * retention))

    distance = _compute_distance_similarity(mutated_df, reference_df, target_column)

    total = _safe_clip(
        (0.45 * perf)
        + (0.25 * logic)
        + (0.20 * intrinsic)
        + (0.10 * distance)
    )

    return ScoreBreakdown(
        total_score=total,
        performance_score=perf,
        logic_score=logic,
        intrinsic_score=intrinsic,
        distance_score=distance,
        macro_f1=macro_f1,
        mcc=mcc,
        type_integrity_score=type_score,
        bound_plausibility_score=bound_score,
        nan_resolution_score=nan_score,
        imbalance_score=imbalance,
        retention_score=retention,
        wasserstein_similarity=distance,
    )