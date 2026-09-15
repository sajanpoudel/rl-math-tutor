# RL-Math

RL-Math trains a PPO agent to pick math word problems for a student.

The agent reads a student's knowledge state, motivation, fatigue, and recent results, then chooses
the next problem to give them. A simulated student attempts that problem using an LLM. The
environment grades the attempt, updates the student, and returns a reward. Over many episodes the
agent learns a problem-selection policy that keeps students in a productive difficulty range.

## Requirements

- Python 3.10
- PostgreSQL 14 or newer with the [pgvector](https://github.com/pgvector/pgvector) extension
- An OpenAI API key, or a local [Ollama](https://ollama.com) server

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gymnasium python-dotenv tqdm textstat tenacity
```

The second `pip install` covers packages the RL environment and the dataset pipeline import at
runtime. If you also plan to use `src/data_processing`, add `spacy` and `datasets`.

## Configuration

Create a `.env` file in the repo root. It is read automatically at startup.

```bash
# Student simulation backend
OPENAI_API_KEY=sk-your-key-here

# Or run against a local Ollama server instead
USE_LOCAL=false
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3.5:9b

# Problem bank database
RLMATH_DATABASE_URL=postgresql://student@localhost:5432/rl_math
```

`OPENAI_API_KEY` is required unless `USE_LOCAL` is `true`, `1`, or `yes`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | none | OpenAI key for student simulation |
| `USE_LOCAL` | `false` | Route student simulation to Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Ollama model name |
| `RLMATH_DATABASE_URL` | `postgresql://student@localhost:5432/rl_math` | Problem bank database |
| `RLMATH_STUDENT_LLM_MODEL` | `gpt-5-nano` | OpenAI model for student simulation |
| `RLMATH_STUDENT_LLM_TIMEOUT` | `45` | Request timeout in seconds |
| `RLMATH_STUDENT_LLM_MAX_RETRIES` | `3` | Retries per LLM call |

## Quick Start

**1. Create the database.**

```bash
createdb rl_math
```

The `pgvector` extension is enabled automatically on first API startup, so the database user needs
permission to run `CREATE EXTENSION`.

**2. Load the problem bank.**

```bash
python -m src.api.import_csv dataset/gsm8k/train_with_concepts_and_subconcepts_stage5.csv
```

This imports 7473 annotated problems. Use `--batch-size` to change the commit batch size.

**3. Start the API.**

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Check it with `curl http://localhost:8000/health`. Interactive docs are at
`http://localhost:8000/docs`.

**4. Train.**

```bash
python scripts/train_and_evaluate.py \
  --api-base-url http://localhost:8000 \
  --timesteps 100000
```

Output is written to `outputs/train_and_eval_<timestamp>/`.

## Project Layout

| Path | Contents |
| --- | --- |
| [src/rl_agent/](src/rl_agent/) | Gymnasium environment, problem selection, reward, evaluation |
| [src/simulation/](src/simulation/) | Simulated students and LLM answer generation |
| [src/api/](src/api/) | Problem bank service over PostgreSQL and pgvector |
| [src/data_processing/](src/data_processing/) | Offline dataset loading and problem bank construction |
| [dataset-pipeline/](dataset-pipeline/) | Annotates raw problems with sub-concepts and difficulty |
| [scripts/](scripts/) | Training and evaluation entry point |
| [dataset/gsm8k/](dataset/gsm8k/) | GSM8K data files |

Each `src` package has its own README with the detail for that folder. The HTTP endpoints are
documented in [api-documentation.md](src/api/api-documentation.md) and
[openapi.yaml](src/api/openapi.yaml).

## The Training Loop

One episode runs up to 12 steps:

1. A student is sampled from seven archetypes with different learning rates, confidence, and
   persistence.
2. The agent emits an 87-dimensional action describing the problem it wants next.
3. The environment matches that action to a real problem by weighted cosine similarity, then applies
   difficulty limits based on how the student is doing.
4. The simulated student decides whether to attempt the problem and produces an answer.
5. The answer is graded, the student's knowledge and motivation update, and a reward is returned.
6. The episode ends after 12 steps, or early if the student disengages.

The agent sees an 81-dimensional observation: a 58-dimensional knowledge vector, 16 context features,
and a 7-dimensional one-hot student type.

## Training Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--api-base-url` | `http://localhost:8000` | Problem bank API |
| `--timesteps` | `100000` | Total training timesteps |
| `--n-envs` | `4` | Parallel environments |
| `--learning-rate` | `1e-4` | PPO learning rate |
| `--n-steps` | `512` | Rollout length per environment |
| `--batch-size` | `64` | Minibatch size |
| `--ent-coef` | `0.02` | Entropy coefficient |
| `--net-arch` | `256 256` | Policy and value network sizes |
| `--max-steps` | `12` | Steps per episode |
| `--num-students` | `200` | Simulated population size |
| `--max-problems` | all | Cap problems loaded from the API |
| `--balanced` | off | Load a difficulty-stratified problem sample |
| `--checkpoint-interval` | `1000` | Timesteps between checkpoints |
| `--eval-episodes` | `100` | Episodes in the final evaluation |
| `--baseline-episodes` | `50` | Episodes per baseline policy |
| `--no-baselines` | off | Skip baseline comparison |
| `--train-seed` / `--test-seed` | `42` / `999` | Training and held-out seeds |
| `--resume-from` | none | Continue from a saved model |
| `--out-dir` | timestamped | Output directory |

A checkpoint is written every `--checkpoint-interval` timesteps into the run directory.

## Dataset

| File | Contents |
| --- | --- |
| `train.csv`, `test.csv` | GSM8K splits, fetched by `download_gsm8k.py` |
| `train_with_concepts_and_subconcepts_stage5.csv` | 7473 annotated problems used by the API |

The annotated file has these columns:

| Column | Contents |
| --- | --- |
| `question`, `answer` | Problem text and worked solution |
| `concepts`, `sub_concepts` | High-level concepts and fine-grained tags |
| `concept_vector_raw` | 58-dimensional intensity vector |
| `concept_vector_norm` | L2-normalized vector, used as the RL state |
| `difficulty_score` | Cognitive load score in `[0.1, 1.0]` |
| `llm_reasoning` | Why those sub-concepts were assigned |

Difficulty scores in this file range from 0.4108 to 0.8905, averaging 0.7163.

To annotate your own problems, put a CSV with `question`, `answer`, and `concepts` columns in
`dataset-pipeline/`, then run:

```bash
export OPENAI_API_KEY="your-key"
python dataset-pipeline/create-dataset.py
```

It reads `gsm8k_processed_stage3.csv` and writes `gsm8k_ready_for_rl_stage5.csv`. The score combines
50 percent intrinsic load from the three heaviest sub-concepts, 30 percent extraneous load from step
count and reading ease, and 20 percent germane load from sub-concept count.

## Concept Vector Layout

All 58 concept dimensions map to these domains:

| Domain | Dimensions |
| --- | --- |
| Arithmetic | 0-9 |
| Fractions and Decimals | 10-17 |
| Percentages and Ratios | 18-23 |
| Algebra | 24-29 |
| Geometry and Measurement | 30-39 |
| Number Theory | 40-43 |
| Statistics and Probability | 44-48 |
| Time and Money | 49-50 |
| Problem Solving Skills | 51-57 |

## Troubleshooting

**`API request failed`** when training starts. The problem bank API is not reachable. Confirm
`uvicorn` is running and `--api-base-url` matches its address.

**`OPENAI_API_KEY must be set`**. Add the key to `.env`, or set `USE_LOCAL=true` to use Ollama.

**Training is slow.** Every step makes at least two LLM calls per student. Reduce `--n-envs`, or set
`USE_LOCAL=true` and point at a local model.

**Empty or tiny problem bank.** Confirm the import succeeded with
`curl http://localhost:8000/api/v1/problems/stats`.
