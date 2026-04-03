from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import AGENT, ALLOWED_BALANCING_ALGOS, DATASET, DOMAIN_RULES, RUNTIME, ensure_runtime_paths
from evaluate import ScoreBreakdown, evaluate_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dataset_health_agent")


TRANSFORM_FALLBACK = '''from __future__ import annotations

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
        raise ValueError(f"Target column '{args.target}' not found.")

    for col in df.columns:
        if col == args.target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
'''


def _load_system_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing system prompt file: {path}")
    return path.read_text(encoding="utf-8")


def _sanitize_generated_code(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""

    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("from ", "import ", "def ", "class ", "if __name__")):
            return "\n".join(text.splitlines()[idx:]).strip()
    return text


def _repair_common_generated_issues(code: str) -> str:
    """Applies minimal safe fixes for frequent LLM mistakes."""

    if not code:
        return code

    lines = code.splitlines()
    import_block: list[str] = []

    has_import_os = any(re.match(r"\s*import\s+os\b", l) for l in lines)
    has_import_sys = any(re.match(r"\s*import\s+sys\b", l) for l in lines)
    has_import_np = any(re.match(r"\s*import\s+numpy\s+as\s+np\b", l) for l in lines)
    has_import_pd = any(re.match(r"\s*import\s+pandas\s+as\s+pd\b", l) for l in lines)

    if re.search(r"\bos\.", code) and not has_import_os:
        import_block.append("import os")
    if re.search(r"\bsys\.", code) and not has_import_sys:
        import_block.append("import sys")
    if re.search(r"\bnp\.", code) and not has_import_np:
        import_block.append("import numpy as np")
    if re.search(r"\bpd\.", code) and not has_import_pd:
        import_block.append("import pandas as pd")

    if not import_block:
        return code

    insert_at = 0
    if lines and lines[0].startswith("from __future__ import"):
        insert_at = 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1

    fixed_lines = lines[:insert_at] + import_block + [""] + lines[insert_at:]
    return "\n".join(fixed_lines).strip() + "\n"


def _call_ollama(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": AGENT.local_model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": AGENT.temperature},
    }

    req = urllib.request.Request(
        AGENT.ollama_host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    max_attempts = 2
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=AGENT.request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
                message = data.get("message", {})
                return str(message.get("content", "")).strip()
        except (TimeoutError, socket.timeout) as exc:
            last_exc = exc
            logger.warning(
                "Ollama request timed out on attempt %s/%s (timeout=%ss)",
                attempt,
                max_attempts,
                AGENT.request_timeout_seconds,
            )
            continue
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {AGENT.ollama_host}") from exc

    raise RuntimeError(
        "Ollama generation timed out after "
        f"{max_attempts} attempts at {AGENT.request_timeout_seconds}s each"
    ) from last_exc


def _call_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    if not AGENT.llm_api_key:
        raise RuntimeError("LLM_API_KEY is required when llm_provider is openai.")

    client = OpenAI(api_key=AGENT.llm_api_key, base_url=AGENT.llm_base_url)
    response = client.chat.completions.create(
        model=AGENT.llm_model,
        temperature=AGENT.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _generate_transform_code(system_prompt: str, user_prompt: str) -> str:
    provider = AGENT.llm_provider.lower().strip()
    if provider == "ollama":
        return _call_ollama(system_prompt, user_prompt)
    if provider in {"openai", "openrouter"}:
        return _call_openai(system_prompt, user_prompt)
    raise RuntimeError(f"Unsupported llm_provider: {AGENT.llm_provider}")


def _extract_algorithms_from_code(code: str) -> list[str]:
    used: list[str] = []
    for algo in ALLOWED_BALANCING_ALGOS:
        if re.search(rf"\b{re.escape(algo)}\b", code):
            used.append(algo)
    return sorted(set(used))


def _read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["?", "NA", "N/A", "none", "null", ""])


def _prepare_working_input(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Prepares a sampler-safe working copy with no NaN in features/target."""

    work = df.copy()

    # Ensure the target does not contain missing values for balancing algorithms.
    work = work[work[target_column].notna()].copy()

    for col in work.columns:
        if col == target_column:
            continue

        if pd.api.types.is_numeric_dtype(work[col]):
            numeric = pd.to_numeric(work[col], errors="coerce")
            median = numeric.median()
            fill_value = float(median) if pd.notna(median) else 0.0
            work[col] = numeric.fillna(fill_value)
        else:
            cleaned = work[col].astype("string")
            mode = cleaned.mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            work[col] = cleaned.fillna(fill_value)

    return work


def _dataset_schema(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": DATASET.target_column,
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "null_counts": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "class_distribution": {
            str(k): int(v)
            for k, v in df[DATASET.target_column].value_counts(dropna=False).to_dict().items()
        }
        if DATASET.target_column in df.columns
        else {},
    }


def _build_user_prompt(
    best_score: ScoreBreakdown,
    schema: dict[str, Any],
    used_algos: list[str],
    previous_error: str | None,
) -> str:
    remaining = [a for a in ALLOWED_BALANCING_ALGOS if a not in used_algos]
    return (
        "Generate full replacement code for transform.py.\n\n"
        f"Current best total score: {best_score.total_score:.6f}\n"
        f"Current best breakdown: {json.dumps(asdict(best_score), indent=2)}\n\n"
        f"Dataset schema: {json.dumps(schema, indent=2)}\n\n"
        f"DOMAIN_RULES: {json.dumps(DOMAIN_RULES, indent=2)}\n\n"
        f"Previously Used Algos: {used_algos}\n"
        f"Remaining Allowed Algos: {remaining}\n\n"
        "Hard requirements:\n"
        "- Output only valid Python code.\n"
        "- Parse --input --output --target.\n"
        "- Use exactly one balancing technique not in Previously Used Algos.\n"
        "- Ensure no unresolved NaN remains in feature columns.\n"
        "- Enforce DOMAIN_RULES before saving output.\n"
        "- If interpolation-based balancing is used, round/cast/clip integer columns.\n\n"
        f"Previous error context: {previous_error or 'None'}\n"
    )


def _run_transform(input_path: Path, output_path: Path, target_column: str) -> tuple[bool, str]:
    # Use --key=value style; argparse accepts this, and custom split('=') parsers also work.
    cmd = [
        sys.executable,
        str(RUNTIME.transform_file),
        f"--input={input_path}",
        f"--output={output_path}",
        f"--target={target_column}",
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(RUNTIME.transform_file.parent),
        capture_output=True,
        text=True,
        timeout=AGENT.transform_timeout_seconds,
        check=False,
    )

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg or "Transform script failed without error output."
    return True, ""


def main() -> None:
    ensure_runtime_paths()

    raw_df = _read_dataset(DATASET.input_raw_dataset)
    if DATASET.target_column not in raw_df.columns:
        raise ValueError(f"Target column '{DATASET.target_column}' not found in input data.")

    working_df = _prepare_working_input(raw_df, DATASET.target_column)
    working_df.to_csv(RUNTIME.raw_working_dataset, index=False)
    RUNTIME.transform_file.write_text(TRANSFORM_FALLBACK, encoding="utf-8")

    baseline = evaluate_dataset(raw_df, raw_df, DATASET.target_column, DOMAIN_RULES)
    best_score = baseline
    best_df = raw_df.copy()

    best_df.to_csv(DATASET.output_best_dataset, index=False)
    best_df.to_csv(RUNTIME.best_dataset_snapshot, index=False)

    logger.info("Baseline health score: %.6f", baseline.total_score)

    system_prompt = _load_system_prompt(RUNTIME.system_prompt_file)
    used_algos: list[str] = []
    previous_error: str | None = None

    for iteration in range(1, AGENT.max_iterations + 1):
        logger.info("Iteration %s/%s", iteration, AGENT.max_iterations)

        schema = _dataset_schema(best_df)
        user_prompt = _build_user_prompt(best_score, schema, used_algos, previous_error)

        try:
            llm_start = pd.Timestamp.now(tz="UTC")
            llm_response = _generate_transform_code(system_prompt, user_prompt)
            llm_elapsed = (pd.Timestamp.now(tz="UTC") - llm_start).total_seconds()
            logger.info("LLM generation completed in %.1f seconds", llm_elapsed)
        except Exception as exc:
            previous_error = f"LLM generation failed: {exc}"
            logger.warning(previous_error)
            continue

        code = _sanitize_generated_code(llm_response)
        code = _repair_common_generated_issues(code)
        if not code:
            previous_error = "Generated code was empty."
            logger.warning(previous_error)
            continue

        algos_in_code = _extract_algorithms_from_code(code)
        non_reused = [a for a in algos_in_code if a not in used_algos]

        if len(non_reused) != 1:
            previous_error = (
                "Code must contain exactly one new allowed balancing algorithm. "
                f"Detected={algos_in_code}, PreviouslyUsed={used_algos}"
            )
            logger.warning(previous_error)
            continue

        selected_algo = non_reused[0]
        used_algos.append(selected_algo)

        generated_file = RUNTIME.generated_code_dir / f"iter_{iteration:02d}_transform.py"
        generated_file.write_text(code, encoding="utf-8")
        RUNTIME.transform_file.write_text(code, encoding="utf-8")
        logger.info("Saved generated transform script: %s", generated_file)
        logger.info("Algorithm used this iteration: %s", selected_algo)

        try:
            ok, err = _run_transform(
                RUNTIME.raw_working_dataset,
                RUNTIME.candidate_dataset,
                DATASET.target_column,
            )
        except subprocess.TimeoutExpired:
            ok = False
            err = (
                "Transform timed out after "
                f"{AGENT.transform_timeout_seconds} seconds."
            )

        # Revert for the next tweak cycle.
        RUNTIME.transform_file.write_text(TRANSFORM_FALLBACK, encoding="utf-8")

        if not ok:
            previous_error = err
            logger.warning("Transform failed: %s", previous_error)
            continue

        candidate_df = _read_dataset(RUNTIME.candidate_dataset)
        candidate_score = evaluate_dataset(candidate_df, raw_df, DATASET.target_column, DOMAIN_RULES)
        logger.info("Candidate total score: %.6f", candidate_score.total_score)

        if candidate_score.total_score > best_score.total_score:
            best_score = candidate_score
            best_df = candidate_df.copy()
            best_df.to_csv(DATASET.output_best_dataset, index=False)
            best_df.to_csv(RUNTIME.best_dataset_snapshot, index=False)
            logger.info("New best score accepted: %.6f", best_score.total_score)
            previous_error = None
        else:
            previous_error = (
                "Candidate did not improve score. "
                f"best={best_score.total_score:.6f}, candidate={candidate_score.total_score:.6f}"
            )

    logger.info("Run complete. Best score: %.6f", best_score.total_score)
    logger.info("Best dataset saved to: %s", DATASET.output_best_dataset)


if __name__ == "__main__":
    main()