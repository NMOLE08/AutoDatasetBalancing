from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
RUNS_DIR: Final[Path] = BASE_DIR / "runs"
PROMPTS_DIR: Final[Path] = BASE_DIR / "prompts"


CLINICAL_SCORE_COLUMNS: Final[tuple[str, ...]] = (
    "erythema",
    "scaling",
    "definite_borders",
    "itching",
    "koebner_phenomenon",
    "polygonal_papules",
    "follicular_papules",
    "oral_mucosal_involvement",
    "knee_and_elbow_involvement",
    "scalp_involvement",
    "family_history",
    "melanin_incontinence",
    "eosinophils_infiltrate",
    "PNL_infiltrate",
    "fibrosis_papillary_dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing_rete_ridges",
    "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis",
    "spongiform_pustule",
    "munro_microabcess",
    "focal_hypergranulosis",
    "disappearance_granular_layer",
    "vacuolisation_damage_basal_layer",
    "spongiosis",
    "saw_tooth_appearance_retes",
    "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate",
)


DOMAIN_RULES: Final[dict[str, dict[str, Any]]] = {
    **{col: {"type": "int", "min": 0, "max": 3} for col in CLINICAL_SCORE_COLUMNS},
    "age": {"type": "int", "min": 0, "max": 120},
    "class": {"type": "int", "min": 1, "max": 6},
}


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
    input_raw_dataset: Path = BASE_DIR / "Dataset" / "dermatology_database_1.csv"
    output_best_dataset: Path = DATA_DIR / "best_dataset.csv"
    target_column: str = "class"


@dataclass(frozen=True)
class AgentConfig:
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")

    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    local_model_name: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:32b")

    max_iterations: int = 2
    transform_timeout_seconds: int = 300
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "900"))
    temperature: float = 0.15


@dataclass(frozen=True)
class RuntimeConfig:
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME.generated_code_dir.mkdir(parents=True, exist_ok=True)