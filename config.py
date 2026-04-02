from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
RUNS_DIR: Final[Path] = BASE_DIR / "runs"
PROMPTS_DIR: Final[Path] = BASE_DIR / "prompts"


ALLOWED_BALANCING_ALGOS: Final[tuple[str, ...]] = (
    "RandomOverSampler",
    "SMOTE",
    "ADASYN",
    "BorderlineSMOTE",
    "KMeansSMOTE",
    "SVMSMOTE",
    "SMOTEN",
    "SMOTENC",
    "RandomUnderSampler",
    "TomekLinks",
    "EditedNearestNeighbours",
    "AllKNN",
    "NearMiss",
    "ClusterCentroids",
    "OneSidedSelection",
    "CondensedNearestNeighbour",
    "NeighbourhoodCleaningRule",
    "InstanceHardnessThreshold",
    "SMOTEENN",
    "SMOTETomek",
    "class_weight",
    "sample_weight",
)


@dataclass(frozen=True)
class DatasetConfig:
    """User-facing dataset locations and schema controls."""

    input_raw_dataset: Path = BASE_DIR / "Dataset" / "dermatology_database_1.csv"
    output_best_dataset: Path = DATA_DIR / "best_dataset.csv"
    target_column: str = "class"
    dataset_modality: str = os.getenv("DATASET_MODALITY", "auto")


@dataclass(frozen=True)
class AgentConfig:
    """User-facing agent controls for iteration and execution safety."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    local_model_name: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")
    auto_pull_local_model: bool = os.getenv("OLLAMA_AUTO_PULL", "1") == "1"
    local_model_pull_timeout_seconds: int = int(os.getenv("OLLAMA_PULL_TIMEOUT_SECONDS", "7200"))
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    max_iterations: int = 25
    transform_timeout_seconds: int = 300
    request_timeout_seconds: int = 240
    temperature: float = 0.25


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime paths used internally by the orchestration loop."""

    transform_file: Path = BASE_DIR / "transform.py"
    raw_working_dataset: Path = RUNS_DIR / "raw_working.csv"
    generated_code_dir: Path = RUNS_DIR / "generated_code"
    candidate_dataset: Path = RUNS_DIR / "candidate.csv"
    best_dataset_snapshot: Path = RUNS_DIR / "best_snapshot.csv"
    system_prompt_file: Path = BASE_DIR / "system_prompt.md"


DATASET = DatasetConfig()
AGENT = AgentConfig()
RUNTIME = RuntimeConfig()


def ensure_runtime_paths() -> None:
    """Creates required directories for stable execution."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME.generated_code_dir.mkdir(parents=True, exist_ok=True)
