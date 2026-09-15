# data_processing

Offline utilities that download raw math datasets and turn them into structured problems with
concept vectors and difficulty scores.

| File | Contents |
| --- | --- |
| `dataset_loaders.py` | `GSM8KLoader`, `OpenMathInstructLoader`, `load_combined_datasets` |
| `problem_bank.py` | `ConceptVectorizer`, `DifficultyAnalyzer`, `ProblemBankCreator` |

## Requirements

This package needs packages beyond the base install:

```bash
pip install spacy textstat datasets requests scikit-learn
```

## Usage

```python
import sys
sys.path.insert(0, "src")

from data_processing.dataset_loaders import load_combined_datasets
from data_processing.problem_bank import ProblemBankCreator

raw = load_combined_datasets(gsm8k_split="train", openmath_max=5_000)

creator = ProblemBankCreator(vector_dim=50)
problems = creator.process_problems(raw)

print(f"{len(problems)} problems")
print(problems[0]["concepts"], problems[0]["difficulty_score"])
```

The result is a list of dicts held in memory. Write it out yourself if you need it persisted.

## Loading Datasets

```python
from data_processing.dataset_loaders import GSM8KLoader, OpenMathInstructLoader

gsm8k = GSM8KLoader(split="train").load()          # or split="test"
openmath = OpenMathInstructLoader(max_samples=10_000).load()
```

Both cache to `./data/cache` by default, changeable with `cache_dir`.

Every loaded problem has:

| Key | Contents |
| --- | --- |
| `id` | Identifier such as `gsm8k_0` |
| `source_dataset` | Which dataset it came from |
| `question` | Problem statement |
| `solution` | Full solution text |
| `answer` | Final numerical answer |
| `solution_steps` | Solution split into steps |
| `step_count` | Number of steps |

`OpenMathInstructLoader` adds `problem_source`, and splits code-interpreter solutions into
`Reasoning:` and `Calculation:` steps.

`load_combined_datasets(gsm8k_split, openmath_max, cache_dir)` loads both and concatenates them.

## Extracting Concepts

```python
from data_processing.problem_bank import ConceptVectorizer

v = ConceptVectorizer(vector_dim=50)
concepts = v.extract_concepts(question, solution)   # list of names
vector = v.create_concept_vector(concepts)          # numpy array, shape (50,)
```

Concepts are matched with regular expressions and written into a 50-dimensional vector:

| Dimensions | Concepts |
| --- | --- |
| 0-7 | addition, subtraction, multiplication, division |
| 8-15 | fractions, decimals, percentages |
| 16-25 | equations, variables, expressions, linear_algebra |
| 26-35 | area, perimeter, volume, angles, coordinate_geometry |
| 36-42 | word_problems, money_problems, time_problems |
| 43-49 | ratios_proportions, statistics_probability, sequences_patterns |

Five names in that table have no matching pattern and are never returned by `extract_concepts`:
`expressions`, `linear_algebra`, `angles`, `coordinate_geometry`, and `sequences_patterns`. The
dimensions they own stay zero.

## Scoring Difficulty

```python
from data_processing.problem_bank import DifficultyAnalyzer

score = DifficultyAnalyzer().calculate_difficulty_score(problem, concepts)
```

Returns a value in `[0.1, 1.0]` from five weighted factors:

| Factor | Weight |
| --- | ---: |
| Solution step count | 25% |
| Concept complexity | 30% |
| Numerical complexity | 20% |
| Reading level | 15% |
| Prerequisite count | 10% |

## Building The Problem Bank

```python
creator = ProblemBankCreator(vector_dim=50)
problems = creator.process_problems(raw_problems)
```

Each processed problem:

| Key | Contents |
| --- | --- |
| `id` | Identifier from the source, or a hash fallback |
| `source_dataset` | Origin dataset |
| `question`, `solution`, `answer` | Cleaned text |
| `concepts` | Detected concept names |
| `concept_vector` | 50-dimensional normalized vector |
| `difficulty_score` | Value in `[0.1, 1.0]` |
| `estimated_time_minutes` | Value in `[1, 30]` |
| `prerequisite_concepts` | Inferred prerequisites |
| `step_count` | Number of solution steps |
| `metadata` | Solution steps, problem source, original record |

Problems with no question or no answer are skipped. A problem that raises is logged and skipped
rather than stopping the batch. A problem where nothing matched falls back to `['addition']`.

Filtering:

```python
easy = creator.get_problems_by_difficulty(problems, max_difficulty=0.3)
algebra = creator.get_problems_by_concepts(problems, ["equations", "variables"])
```

`get_problems_by_concepts` keeps a problem if it has any of the listed concepts, not all of them.
