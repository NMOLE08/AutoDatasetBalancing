from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import AGENT, ALLOWED_BALANCING_ALGOS, DATASET, RUNTIME, ensure_runtime_paths
from evaluate import ScoreBreakdown, evaluate_dataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("autonomous_dataset_balancer")


TRANSFORM_TEMPLATE = '''from __future__ import annotations

import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in input dataset.")

    # Default fallback: lightweight cleanup only.
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    for col in df.columns:
        if col == args.target:
            continue
        if df[col].dtype.kind in {"i", "u", "f"}:
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
'''


def _load_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"System prompt file is missing: {path}")
    return path.read_text(encoding="utf-8")


def _read_tabular_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _detect_dataset_modality(path: Path) -> str:
    configured = DATASET.dataset_modality.lower().strip()
    if configured in {"tabular", "image"}:
        return configured

    if path.is_dir():
        has_images = any(_is_image_file(p) for p in path.rglob("*"))
        if has_images:
            return "image"
        return "tabular"

    suffix = path.suffix.lower()
    if suffix in {".csv", ".json", ".jsonl", ".parquet", ".tsv"}:
        return "tabular"
    if _is_image_file(path):
        return "image"
    return "tabular"


def _load_image_dataset_as_tabular(path: Path, target_column: str) -> pd.DataFrame:
    try:
        from PIL import Image, ImageStat
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image dataset ingestion. Install with: pip install pillow"
        ) from exc

    image_paths: list[Path] = []
    if path.is_dir():
        image_paths = [p for p in path.rglob("*") if p.is_file() and _is_image_file(p)]
    elif path.is_file() and _is_image_file(path):
        image_paths = [path]

    if not image_paths:
        raise ValueError(f"No image files found in {path}")

    rows: list[dict[str, Any]] = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                rgb = img.convert("RGB")
                stat = ImageStat.Stat(rgb)
                channels = len(rgb.getbands())
                mean_intensity = float(sum(stat.mean) / max(len(stat.mean), 1))
                std_intensity = float(sum(stat.stddev) / max(len(stat.stddev), 1))
                width, height = rgb.size
        except Exception:
            # Skip corrupted files; they are low-quality input for balancing.
            continue

        label = img_path.parent.name if path.is_dir() else "unknown"
        rows.append(
            {
                "image_path": str(img_path),
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "file_size_bytes": int(img_path.stat().st_size),
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                target_column: label,
            }
        )

    if not rows:
        raise ValueError("Unable to parse any valid images from the input path.")

    return pd.DataFrame(rows)


def _flatten_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "meta" in work.columns:
        if work["meta"].apply(lambda v: isinstance(v, dict)).all():
            meta_df = pd.json_normalize(work["meta"])
            work = work.drop(columns=["meta"]).join(meta_df.add_prefix("meta."))

    return work


def _serialize_complex_columns(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    work = df.copy()

    # Keep list/dict columns serializable and model-safe.
    for col in work.columns:
        if col == target_column:
            continue
        if work[col].apply(lambda v: isinstance(v, (dict, list))).any():
            work[col] = work[col].apply(
                lambda v: json.dumps(v, ensure_ascii=True) if isinstance(v, (dict, list)) else v
            )

    return work


def _impute_missing_values(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    work = df.copy()
    for col in work.columns:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(work[col]):
            median = work[col].median()
            fill_value = float(median) if pd.notna(median) else 0.0
            work[col] = work[col].fillna(fill_value)
        else:
            mode = work[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            work[col] = work[col].fillna(fill_value)

    return work


def _encode_categorical_features(
    df: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
    """Deterministically encodes object/category columns for balancers requiring numeric features."""

    work = df.copy()
    encoders: dict[str, dict[int, str]] = {}
    for col in work.columns:
        if col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            continue

        normalized = work[col].astype(str).fillna("unknown")
        codes, uniques = pd.factorize(normalized, sort=True)
        work[col] = codes.astype("int64")
        encoders[col] = {int(idx): str(label) for idx, label in enumerate(uniques.tolist())}

    return work, encoders


def _decode_for_output(
    df: pd.DataFrame,
    encoders: dict[str, dict[int, str]],
    target_column: str,
) -> pd.DataFrame:
    """Converts encoded features back to readable categories where mapping is available."""

    out = df.copy()

    for col, mapping in encoders.items():
        if col not in out.columns:
            continue

        def _decode_value(value: Any) -> Any:
            if pd.isna(value):
                return value
            try:
                value_f = float(value)
                value_i = int(round(value_f))
                if abs(value_f - value_i) < 1e-9:
                    return mapping.get(value_i, value)
                return value
            except (TypeError, ValueError):
                return value

        out[col] = out[col].map(_decode_value)

    # Remove unnecessary .0 formatting for integer-like float columns.
    for col in out.columns:
        if col == target_column:
            continue
        if pd.api.types.is_float_dtype(out[col]):
            series = out[col]
            non_null = series.dropna()
            if not non_null.empty and non_null.map(lambda v: abs(v - round(v)) < 1e-9).all():
                out[col] = series.round().astype("Int64")

    return out


def _prune_high_cardinality_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    work = df.copy()
    to_drop: list[str] = []

    for col in work.columns:
        if col == target_column:
            continue

        series = work[col]
        if series.empty:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_ratio = float(non_null.nunique(dropna=True)) / float(len(non_null))

        if pd.api.types.is_numeric_dtype(series):
            continue

        as_str = non_null.astype(str)
        mean_len = float(as_str.str.len().mean())

        # Remove identifier-like or free-text columns that create sparse one-hot blowups.
        if unique_ratio > 0.95 and mean_len > 12:
            to_drop.append(col)
            continue

        if mean_len > 240:
            to_drop.append(col)

    if to_drop:
        logger.info("Dropping high-cardinality/text columns: %s", to_drop)
        work = work.drop(columns=to_drop)

    return work


def _infer_target_column(df: pd.DataFrame) -> pd.Series:
    if "meta.cwe" in df.columns:
        return df["meta.cwe"].apply(
            lambda v: (v[0] if isinstance(v, list) and len(v) > 0 else "Unknown")
        )
    if "meta.vulnerability_type" in df.columns:
        return df["meta.vulnerability_type"].astype(str).replace("", "Unknown")
    if "meta.task" in df.columns:
        return df["meta.task"].astype(str).replace("", "Unknown")
    if "id" in df.columns:
        return df["id"].astype(str)
    return pd.Series(["Unknown"] * len(df), index=df.index)


def _load_raw_dataset() -> pd.DataFrame:
    if not DATASET.input_raw_dataset.exists():
        raise FileNotFoundError(
            f"Input dataset was not found at {DATASET.input_raw_dataset}. "
            "Please place your dataset there or update config.py."
        )

    modality = _detect_dataset_modality(DATASET.input_raw_dataset)
    if modality == "image":
        logger.info("Detected image dataset. Extracting lightweight tabular features.")
        df = _load_image_dataset_as_tabular(DATASET.input_raw_dataset, DATASET.target_column)
    else:
        df = _read_tabular_dataset(DATASET.input_raw_dataset)
        df = _flatten_nested_columns(df)

    if DATASET.target_column not in df.columns:
        df[DATASET.target_column] = _infer_target_column(df)

    df = _serialize_complex_columns(df, DATASET.target_column)
    df = _impute_missing_values(df, DATASET.target_column)
    df = _prune_high_cardinality_features(df, DATASET.target_column)

    if df.empty:
        raise ValueError("Input dataset is empty.")
    if DATASET.target_column not in df.columns:
        raise ValueError(
            f"Target column '{DATASET.target_column}' does not exist in input dataset."
        )
    return df


def _dataset_schema(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "detected_modality": _detect_dataset_modality(DATASET.input_raw_dataset),
        "target_column": DATASET.target_column,
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "class_distribution": df[DATASET.target_column].value_counts(dropna=False).to_dict(),
        "null_counts": df.isna().sum().to_dict(),
    }


def _build_user_prompt(
    baseline: ScoreBreakdown,
    schema: dict[str, Any],
    used_algos: list[str],
    candidate_algos: list[str],
    previous_error: str | None,
) -> str:
    return (
        "Generate replacement Python code for transform.py.\n\n"
        f"Baseline total score: {baseline.total_score:.6f}\n"
        f"Baseline breakdown: {json.dumps(asdict(baseline), indent=2)}\n\n"
        f"Dataset schema: {json.dumps(schema, indent=2)}\n\n"
        f"Previously Used Algos: {used_algos}\n\n"
        f"Useful Candidate Algos This Iteration: {candidate_algos}\n\n"
        "Constraints:\n"
        "- Output only valid Python code.\n"
        "- Must parse --input --output --target.\n"
        "- Must save transformed CSV to --output.\n"
        "- Must use at most one balancing algorithm.\n"
        "- Must not reuse previously used algorithms.\n"
        "- Prefer the most suitable algorithm from Useful Candidate Algos This Iteration.\n"
        "- Never raise or fail only because input has NaN values.\n"
        "- Impute/fill missing values and ensure no NaN remains in feature columns before saving.\n\n"
        f"Previous error context: {previous_error or 'None'}\n"
    )


def _sanitize_generated_code(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return text

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # If model prepends prose, keep the most likely code section starting from imports/defs.
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("from ", "import ", "def ", "class ", "if __name__")):
            return _repair_common_import_paths("\n".join(lines[idx:]).strip())

    return _repair_common_import_paths(text)


def _repair_common_import_paths(code: str) -> str:
    """Fixes common imbalanced-learn import mistakes emitted by LLMs."""

    fixed = code

    # Hybrid combiners live under imblearn.combine, not over_sampling.
    fixed = re.sub(
        r"from\s+imblearn\.over_sampling\s+import\s+([^\n#]*\bSMOTEENN\b[^\n#]*)",
        lambda m: (
            m.group(0).replace("from imblearn.over_sampling import", "from imblearn.combine import")
        ),
        fixed,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r"from\s+imblearn\.over_sampling\s+import\s+([^\n#]*\bSMOTETomek\b[^\n#]*)",
        lambda m: (
            m.group(0).replace("from imblearn.over_sampling import", "from imblearn.combine import")
        ),
        fixed,
        flags=re.IGNORECASE,
    )

    # Boundary cleaners live under imblearn.under_sampling.
    fixed = re.sub(
        r"from\s+imblearn\.over_sampling\s+import\s+([^\n#]*\bTomekLinks\b[^\n#]*)",
        lambda m: (
            m.group(0).replace("from imblearn.over_sampling import", "from imblearn.under_sampling import")
        ),
        fixed,
        flags=re.IGNORECASE,
    )
    fixed = re.sub(
        r"from\s+imblearn\.over_sampling\s+import\s+([^\n#]*\bEditedNearestNeighbours\b[^\n#]*)",
        lambda m: (
            m.group(0).replace("from imblearn.over_sampling import", "from imblearn.under_sampling import")
        ),
        fixed,
        flags=re.IGNORECASE,
    )

    return fixed


def _ensure_local_ollama_model() -> None:
    if AGENT.llm_provider.lower() != "ollama":
        return

    try:
        listed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama is not installed or not available in PATH.") from exc

    if listed.returncode != 0:
        raise RuntimeError(f"Failed to list Ollama models: {(listed.stderr or '').strip()}")

    if AGENT.local_model_name in listed.stdout:
        logger.info("Local model already available in Ollama: %s", AGENT.local_model_name)
        return

    if not AGENT.auto_pull_local_model:
        raise RuntimeError(
            f"Model '{AGENT.local_model_name}' is not pulled. "
            "Enable OLLAMA_AUTO_PULL=1 or pull manually with: "
            f"ollama pull {AGENT.local_model_name}"
        )

    logger.info("Pulling local model once: %s", AGENT.local_model_name)
    pulled = subprocess.run(
        ["ollama", "pull", AGENT.local_model_name],
        capture_output=True,
        text=True,
        timeout=AGENT.local_model_pull_timeout_seconds,
        check=False,
    )
    if pulled.returncode != 0:
        raise RuntimeError(
            f"Ollama pull failed for '{AGENT.local_model_name}'. "
            f"STDERR: {(pulled.stderr or '').strip()}"
        )


def _generate_transform_code_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": AGENT.local_model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": AGENT.temperature,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    chat_url = AGENT.ollama_host.rstrip("/") + "/api/chat"
    request = urllib.request.Request(
        chat_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=AGENT.request_timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama request timed out after {AGENT.request_timeout_seconds} seconds."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Ollama at {chat_url}: {exc}") from exc

    parsed = json.loads(raw)
    content = parsed.get("message", {}).get("content", "")
    if not content.strip():
        raise RuntimeError("Ollama returned an empty transform script.")

    return _sanitize_generated_code(content)


def _generate_transform_code_openai(system_prompt: str, user_prompt: str) -> str:
    if not AGENT.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY is not configured. Set LLM_API_KEY or use LLM_PROVIDER=ollama."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for OpenAI-compatible calls. Install with: pip install openai"
        ) from exc

    client = OpenAI(api_key=AGENT.llm_api_key, base_url=AGENT.llm_base_url)
    response = client.chat.completions.create(
        model=AGENT.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=AGENT.temperature,
        timeout=AGENT.request_timeout_seconds,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("LLM returned an empty transform script.")

    return _sanitize_generated_code(content)


def _generate_transform_code(system_prompt: str, user_prompt: str) -> str:
    provider = AGENT.llm_provider.lower().strip()
    if provider == "ollama":
        return _generate_transform_code_ollama(system_prompt, user_prompt)
    if provider in {"openai", "openrouter"}:
        return _generate_transform_code_openai(system_prompt, user_prompt)
    raise RuntimeError(f"Unsupported LLM provider: {AGENT.llm_provider}")


def _extract_algo_used(code: str) -> str | None:
    for algo in ALLOWED_BALANCING_ALGOS:
        pattern = re.compile(rf"\b{re.escape(algo)}\b", flags=re.IGNORECASE)
        if pattern.search(code):
            return algo
    return None


def _validate_generated_code(
    code: str,
    used_algos: list[str],
    candidate_algos: list[str],
    has_categorical_features: bool,
) -> str | None:
    """Returns None when code is valid, otherwise a user-feedback error string."""

    if "argparse" not in code or "--input" not in code or "--output" not in code or "--target" not in code:
        return "Generated code must parse --input, --output, and --target with argparse."

    if "to_csv" not in code:
        return "Generated code must save transformed data with to_csv(output)."

    algo_used = _extract_algo_used(code)
    has_remaining_allowed_algos = len(used_algos) < len(ALLOWED_BALANCING_ALGOS)
    has_remaining_candidate_algos = len(candidate_algos) > 0
    if has_remaining_allowed_algos and has_remaining_candidate_algos and algo_used is None:
        return (
            "Generated code did not use any useful balancing algorithm for this iteration. "
            f"Choose one of: {candidate_algos}"
        )

    if "fit_resample(" in code and algo_used is None:
        return (
            "Generated code uses fit_resample with a non-allowed balancer. "
            f"Only these are allowed: {list(ALLOWED_BALANCING_ALGOS)}"
        )

    if algo_used is not None and algo_used.lower() == "smotenc" and not has_categorical_features:
        return (
            "SMOTENC was selected but the current working dataset has no categorical feature columns. "
            "Choose SMOTE/ADASYN/RandomOverSampler or another compatible strategy."
        )

    if algo_used is not None and algo_used not in candidate_algos and has_remaining_candidate_algos:
        return (
            f"Algorithm '{algo_used}' is not considered useful for this dataset iteration. "
            f"Choose one of: {candidate_algos}"
        )

    return None


def _compute_candidate_algos(
    df: pd.DataFrame,
    target_column: str,
    used_algos: list[str],
    has_categorical_features: bool,
) -> list[str]:
    """Returns dataset-aware useful algorithms that remain unused."""

    remaining = [algo for algo in ALLOWED_BALANCING_ALGOS if algo not in used_algos]
    if not remaining:
        return []

    counts = df[target_column].value_counts(dropna=False)
    if counts.empty:
        return []

    imbalance_ratio = float(counts.min() / max(counts.max(), 1))

    # If classes are already close to balanced, additional balancing is likely not useful.
    if imbalance_ratio >= 0.95:
        return []

    candidates = list(remaining)

    # SMOTENC requires categorical features in the working dataset.
    if not has_categorical_features:
        candidates = [algo for algo in candidates if algo not in {"SMOTENC", "SMOTEN"}]

    # Very small datasets are fragile for high-complexity generators.
    if len(df) < 40:
        candidates = [
            algo
            for algo in candidates
            if algo not in {"KMeansSMOTE", "SVMSMOTE", "InstanceHardnessThreshold"}
        ]

    return candidates


def _write_transform(code: str) -> None:
    RUNTIME.transform_file.write_text(code + "\n", encoding="utf-8")


def _save_generated_transform(iteration: int, code: str) -> Path:
    output_file = RUNTIME.generated_code_dir / f"iter_{iteration:02d}_transform.py"
    output_file.write_text(code + "\n", encoding="utf-8")
    return output_file


def _reset_transform() -> None:
    _write_transform(TRANSFORM_TEMPLATE)


def _run_transform(input_path: Path, output_path: Path, target_column: str) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(RUNTIME.transform_file),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--target",
        target_column,
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=AGENT.transform_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Transform timed out after {AGENT.transform_timeout_seconds} seconds."
    except Exception as exc:
        return False, f"Transform crashed before completion: {exc}"

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        return False, f"Transform failed with code {completed.returncode}. STDERR: {stderr} STDOUT: {stdout}"

    if not output_path.exists():
        return False, "Transform succeeded but did not create an output CSV."

    return True, "Transform completed successfully."


def _try_read_candidate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate dataset not found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Candidate dataset is empty.")
    if DATASET.target_column not in df.columns:
        raise ValueError(
            f"Candidate dataset is missing required target column '{DATASET.target_column}'."
        )
    return df


def run() -> None:
    ensure_runtime_paths()
    _ensure_local_ollama_model()
    if AGENT.llm_provider.lower() == "ollama":
        logger.info("LLM provider: ollama | host: %s | model: %s", AGENT.ollama_host, AGENT.local_model_name)
    else:
        logger.info("LLM provider: %s | model: %s", AGENT.llm_provider, AGENT.llm_model)

    logger.info("Input dataset: %s | target: %s", DATASET.input_raw_dataset, DATASET.target_column)
    system_prompt = _load_system_prompt(RUNTIME.system_prompt_file)

    raw_df = _load_raw_dataset()
    has_categorical_features = any(
        col != DATASET.target_column and not pd.api.types.is_numeric_dtype(raw_df[col])
        for col in raw_df.columns
    )
    working_df, encoders = _encode_categorical_features(raw_df, DATASET.target_column)

    working_df.to_csv(RUNTIME.raw_working_dataset, index=False)
    _decode_for_output(working_df, encoders, DATASET.target_column).to_csv(
        DATASET.output_best_dataset,
        index=False,
    )

    baseline = evaluate_dataset(working_df, DATASET.target_column, working_df)
    logger.info("Baseline score: %.6f", baseline.total_score)

    best_overall_score = baseline.total_score
    best_dataset_df = working_df.copy()

    used_algos: list[str] = []
    previous_error: str | None = None

    _reset_transform()

    iteration = 1
    while iteration <= AGENT.max_iterations:
        logger.info("Starting iteration %d/%d", iteration, AGENT.max_iterations)
        schema = _dataset_schema(working_df)
        candidate_algos = _compute_candidate_algos(
            working_df,
            DATASET.target_column,
            used_algos,
            has_categorical_features,
        )

        if not candidate_algos:
            logger.info(
                "Ending early: no useful/remaining balancing algorithms for current dataset state."
            )
            break

        prompt = _build_user_prompt(
            baseline=baseline,
            schema=schema,
            used_algos=used_algos,
            candidate_algos=candidate_algos,
            previous_error=previous_error,
        )

        try:
            candidate_code = _generate_transform_code(system_prompt=system_prompt, user_prompt=prompt)
            saved_script = _save_generated_transform(iteration, candidate_code)
            logger.info("Saved generated transform script: %s", saved_script)

            validation_error = _validate_generated_code(
                candidate_code,
                used_algos,
                candidate_algos,
                has_categorical_features,
            )
            if validation_error:
                previous_error = validation_error
                logger.warning("Skipping invalid generated code: %s", validation_error)
                iteration += 1
                continue

            algo_used = _extract_algo_used(candidate_code)

            if algo_used is not None and algo_used in used_algos:
                previous_error = (
                    f"Generated code attempted to reuse forbidden algorithm '{algo_used}'. "
                    "Pick a new balancing algorithm."
                )
                logger.warning(previous_error)
                iteration += 1
                continue

            if algo_used is not None:
                used_algos.append(algo_used)
                logger.info("Algorithm used this iteration: %s", algo_used)
            else:
                logger.info("No known balancing algorithm detected in generated code.")

            _write_transform(candidate_code)
            success, message = _run_transform(
                input_path=RUNTIME.raw_working_dataset,
                output_path=RUNTIME.candidate_dataset,
                target_column=DATASET.target_column,
            )
            logger.info(message)

            if not success:
                previous_error = message
                iteration += 1
                continue

            candidate_df = _try_read_candidate(RUNTIME.candidate_dataset)
            candidate_score = evaluate_dataset(candidate_df, DATASET.target_column, working_df)
            logger.info("Candidate total score: %.6f", candidate_score.total_score)

            if candidate_score.total_score > best_overall_score:
                best_overall_score = candidate_score.total_score
                best_dataset_df = candidate_df.copy()
                _decode_for_output(best_dataset_df, encoders, DATASET.target_column).to_csv(
                    DATASET.output_best_dataset,
                    index=False,
                )
                best_dataset_df.to_csv(RUNTIME.best_dataset_snapshot, index=False)
                logger.info("New best score accepted: %.6f", best_overall_score)

            previous_error = None

        except Exception as exc:
            previous_error = f"Iteration failed: {exc}"
            logger.exception(previous_error)

        finally:
            # Roll back playground for the next fresh transform attempt.
            _reset_transform()

        iteration += 1

    _decode_for_output(best_dataset_df, encoders, DATASET.target_column).to_csv(
        DATASET.output_best_dataset,
        index=False,
    )
    logger.info("Run complete. Best score: %.6f", best_overall_score)
    logger.info("Best dataset saved to: %s", DATASET.output_best_dataset)


if __name__ == "__main__":
    run()
