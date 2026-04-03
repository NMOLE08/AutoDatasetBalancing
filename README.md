# AutoDatasetBalancing

Autonomous dataset balancing pipeline for tabular clinical-style datasets.

The project uses an LLM (default: Ollama with qwen2.5-coder:32b) to iteratively generate balancing transforms, execute them, evaluate quality, and keep the best candidate.

## Features

- Iterative balancing loop with LLM-generated transform code
- Composite Dataset Health Score optimization
- Strict domain integrity checks (integer type, min/max bounds)
- Additional penalties for unresolved missing markers
- Safe execution pipeline with transform timeout, retry, and script repair for common LLM mistakes

## Project Structure

- [config.py](config.py): user-facing settings, paths, domain rules, model/timeout settings
- [evaluate.py](evaluate.py): Dataset Health Score evaluator
- [main.py](main.py): orchestration loop
- [system_prompt.md](system_prompt.md): LLM behavior and generation constraints
- [transform.py](transform.py): temporary generated script per iteration
- [runs/](runs/): run artifacts and generated transform scripts
- [data/](data/): output datasets (ignored in git)
- [Dataset/](Dataset/): raw input datasets (ignored in git)

## Scoring Formula

The evaluator optimizes:

Total_Score = (0.45 * Performance) + (0.25 * Logic) + (0.20 * Intrinsic) + (0.10 * Distance)

Where:

- Performance:
  - Macro F1 + MCC
- Logic:
  - Type integrity (illegal fractions in int-ruled columns)
  - Bound plausibility (min/max violations)
  - NaN resolution
  - Extra severe penalties:
    - `-0.20` if exact `?` appears anywhere
    - `-0.10` if unresolved `np.nan` remains at evaluation time
- Intrinsic:
  - Class imbalance ratio
  - Retention score
- Distance:
  - Wasserstein-based similarity

## Requirements

- Python 3.13+
- Ollama running locally (if using default provider)
- Model pulled in Ollama (default: `qwen2.5-coder:32b`)

Python packages used:

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- openai (only if using OpenAI/OpenRouter provider)
- pillow

## Setup

From the repository root:

```powershell
cd C:\Users\ouro\V2_Dataset_balancing\AutoDatasetBalancing

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn imbalanced-learn openai pillow
```

## Configure

Edit [config.py](config.py):

- `DATASET.input_raw_dataset`
- `DATASET.target_column`
- `DOMAIN_RULES`
- `AGENT.llm_provider` (`ollama` by default)
- `AGENT.local_model_name`
- `AGENT.max_iterations`
- `AGENT.request_timeout_seconds`
- `AGENT.transform_timeout_seconds`

## Run

```powershell
cd C:\Users\ouro\V2_Dataset_balancing\AutoDatasetBalancing
.\.venv\Scripts\Activate.ps1
python .\main.py
```

## Output

- Best dataset: [data/best_dataset.csv](data/best_dataset.csv)
- Best snapshot: [runs/best_snapshot.csv](runs/best_snapshot.csv)
- Candidate each iteration: [runs/candidate.csv](runs/candidate.csv)
- Generated scripts per iteration: [runs/generated_code/](runs/generated_code/)

## Troubleshooting

- Long pauses at iteration start:
  - Usually LLM generation latency. Check Ollama responsiveness.
- LLM generation timed out:
  - Increase `REQUEST_TIMEOUT_SECONDS` env var or `request_timeout_seconds` in [config.py](config.py).
- Transform failures from generated code:
  - The pipeline already retries and applies common repairs, then moves to next iteration.
- Wrong Python environment:
  - Ensure you activate [AutoDatasetBalancing/.venv](AutoDatasetBalancing/.venv) before running.

## Git Notes

The following are intentionally ignored in git:

- `Dataset/`
- `data/`
- `.venv/`

This keeps raw and generated datasets out of source control by default.
