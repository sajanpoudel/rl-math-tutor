# simulation

The simulated student used by the tutoring environment. Given a problem, a student decides whether to
attempt it, produces an answer through an LLM, and updates its own cognitive state from the result.

| File | Contents |
| --- | --- |
| `student_model.py` | `CognitiveStudentModel`, `StudentPopulation`, knowledge graph and state updates |
| `student_agents.py` | `DecisionAgent` and `SolverAgent` for OpenAI and Ollama |
| `llm_answer_generator.py` | Backend configuration and agent construction |

## Usage

```python
import sys
sys.path.insert(0, "src")

from simulation import StudentPopulation

population = StudentPopulation(num_students=200, concept_dim=58)
student = population.sample_student()

result = student.attempt_problem({
    "id": 1,
    "question": "Sarah has 3 boxes with 12 pencils each. How many pencils?",
    "answer": "36",
    "concept_vector": [0.0] * 58,
    "difficulty_score": 0.45,
    "concepts": ["multiplication"],
    "sub_concepts": ["mul_fact_retrieval"],
})

print(result["success"], result["final_answer"], result["gave_up"])
```

`sample_student()` builds a new student from the archetype distribution. `get_random_student()`
returns one from the pre-built pool instead.

A backend must be configured before calling `attempt_problem`, since every attempt makes LLM calls.

## Configuration

Set these in a `.env` file at the repo root or in the environment.

OpenAI:

```bash
OPENAI_API_KEY=sk-your-key-here
```

Ollama:

```bash
USE_LOCAL=true
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen3.5:9b
```

`USE_LOCAL` counts as on when set to `true`, `1`, or `yes`. `OPENAI_API_KEY` is required when it is
off.

| Setting | OpenAI | Ollama |
| --- | --- | --- |
| Model | `gpt-5-nano` | `qwen3.5:9b` |
| Max output tokens | 1200 | 6000 |
| Timeout | 45s | 120s |
| Retries | 3 | 3 |
| Refinement rounds and beam size | 3 and 2 | 1 and 1 |

Overrides for both backends: `RLMATH_STUDENT_LLM_MAX_OUTPUT_TOKENS`, `RLMATH_STUDENT_LLM_TIMEOUT`,
`RLMATH_STUDENT_LLM_MAX_RETRIES`. OpenAI also accepts `RLMATH_STUDENT_LLM_MODEL`,
`RLMATH_STUDENT_LLM_REASONING_EFFORT`, and `RLMATH_STUDENT_LLM_VERBOSITY`.

## Student Types

| Type | Share | Behavior |
| --- | ---: | --- |
| `average` | 30% | Balanced learning rate, confidence, and persistence |
| `analytical` | 15% | Step-by-step, steady, few careless errors |
| `intuitive` | 15% | Fast pattern matching, more variable reasoning |
| `fast_learner` | 15% | Higher learning rate and larger gains |
| `struggling_learner` | 15% | Lower starting knowledge, more mistakes |
| `anxious_learner` | 5% | Low confidence, sensitive to setbacks |
| `confident_learner` | 5% | High confidence and persistence |

Each type has its own decision and solver prompts, so two types given the same problem and the same
knowledge state behave differently. `analytical` and `intuitive` have dedicated subclasses,
`AnalyticalStudentModel` and `IntuitiveStudentModel`.

To build one type directly:

```python
from simulation import AnalyticalStudentModel

student = AnalyticalStudentModel(student_id="s1", concept_dim=58)
```

## What A Student Tracks

- A 58-dimensional knowledge vector, updated in logit space
- A knowledge graph of concept and sub-concept nodes with good and bad evidence counts
- Motivation, fatigue, and confidence
- Consecutive successes and failures
- Recent results, attempt history, and mistake patterns

Read it with `student.get_current_state()`, `student.get_context_features(16)`, or
`student.get_full_knowledge_graph()`. Call `student.reset_session()` to start a new session while
keeping learned knowledge.

Node mastery is `(good + 1) / (good + bad + 2)`, so an unseen concept starts at 0.65.

## What An Attempt Does

1. Builds a profile of the problem: active concepts, mastery, and similar past attempts.
2. Estimates success probability from mastery and difficulty, adjusted by motivation, fatigue,
   attention, and any hint, clamped to `[0.05, 0.95]`.
3. `DecisionAgent` returns `can_solve`, `confidence`, `predicted_outcome`, `error_types`,
   `practice_targets`, and `reasoning`.
4. `SolverAgent` returns `answer_text`, `final_answer`, `gave_up`, `help_requested`,
   `needs_more_training`, `hint_text`, and `practice_targets`.
5. Beam refinement runs until alignment reaches 0.9 or the rounds are used up.
6. The answer is graded numerically when it can be parsed.
7. State updates.

`attempt_problem` returns a dict with `success`, `gave_up`, `final_answer`, `answer_correct`,
`answer_was_graded`, `confidence`, `help_requested`, `needs_more_training`, `mistake_penalty`,
`time_spent`, and `practice_targets`.

## How Results Change Knowledge

| Outcome | Evidence | Knowledge vector |
| --- | --- | --- |
| Correct | Good, weight 1.0 | Positive update |
| Wrong but attempted | Bad, weight 0.5 | Partial negative update |
| Gave up | Bad, weight 1.0 | No update |

An attempted wrong answer still teaches something. Giving up does not.

## Population

```python
population = StudentPopulation(num_students=200, concept_dim=58)

population.sample_student()        # new student from the type distribution
population.get_random_student()    # existing student from the pool
population.get_population_stats()  # type counts and knowledge summary
```
