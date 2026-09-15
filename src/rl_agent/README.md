# rl_agent

The reinforcement-learning environment, the problem bank client, and the evaluation framework.

| File | Contents |
| --- | --- |
| `tutoring_env.py` | `SimulatedTutoringEnv`, the Gymnasium environment |
| `problem_bank.py` | `ProblemBank` and `ProblemRecord`, loaded from the API |
| `evaluation_framework.py` | `evaluate_policy`, baseline policies, metrics and logs |

## Usage

The environment needs a running problem bank API. Import from the `src` directory:

```python
import sys
sys.path.insert(0, "src")

from rl_agent.tutoring_env import SimulatedTutoringEnv

env = SimulatedTutoringEnv(api_base_url="http://localhost:8000")

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

`SimulatedTutoringEnv` accepts:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `api_base_url` | `http://localhost:8000` | Problem bank API |
| `max_problems` | `None` | Cap on problems to load |
| `max_steps` | `12` | Steps per episode |
| `num_students` | `200` | Simulated population size |
| `seed` | `None` | Seeds Python and NumPy RNGs |
| `page_size` | `500` | Problems fetched per API request |
| `min_difficulty`, `max_difficulty` | `None` | Restrict the loaded difficulty range |
| `concepts`, `sub_concepts` | `None` | Restrict loaded problems by tag |
| `balanced` | `False` | Load a difficulty-stratified sample instead of everything |

`reset()` returns `(observation, info)` and `step()` returns
`(observation, reward, terminated, truncated, info)`. The `info` dict from `step()` carries
`problem_id`, `difficulty`, `success`, `gave_up`, `final_answer`, `answer_correct`,
`answer_was_graded`, `knowledge_delta`, `motivation_delta`, `confidence`, `motivation`,
`student_type`, and `needs_more_training`.

To run a specific student instead of a random one, pass it through `reset`:

```python
obs, info = env.reset(options={"probe_student": student})
```

## Observation Space

`Box(0.0, 1.0, shape=(81,), dtype=float32)`.

| Slice | Size | Contents |
| --- | ---: | --- |
| `0:58` | 58 | Per-concept mastery |
| `58:74` | 16 | Context features |
| `74:81` | 7 | Student type one-hot |

The 16 context features, in order: motivation, fatigue, consecutive failures over 10, recent success
rate, session problems over 20, recent answer accuracy, last answer graded, last answer correct, last
help requested, last needs-more-training, last gave up, last confidence, graph nodes over 300, graph
edges over 600, knowledge nodes over 300, consecutive successes over 10. All clamped to `[0, 1]`.

The seven student types, in one-hot order: `average`, `analytical`, `intuitive`, `fast_learner`,
`struggling_learner`, `anxious_learner`, `confident_learner`.

## Action Space

`Box(0.0, 1.0, shape=(87,), dtype=float32)`.

| Slice | Size | Contents |
| --- | ---: | --- |
| `0:58` | 58 | Target concept emphasis |
| `58:87` | 29 | Target problem metadata |

The 29 metadata dimensions:

| Slot | Contents |
| --- | --- |
| 0-4 | Difficulty, concept count, sub-concept count, prerequisite count, estimated minutes |
| 5-7 | Question length, solution length, solution step count |
| 8-16 | One binary flag per concept domain |
| 17-19 | Concept sparsity, dominant concept strength, answer magnitude |
| 20-28 | Continuous strength per concept domain |

## Problem Selection

The action is a description of the problem you want, not an index. Each step the environment scores
every problem in the bank and samples one:

1. Concept dimensions are scaled by 2.0, then the action is L2-normalized.
2. Scores come from the dot product against every problem's normalized action vector.
3. After a wrong answer or give-up, problems covering the failed concepts get a boost, weighted 0.30
   after a wrong answer and 0.60 after a give-up.
4. After two or more consecutive failures, easier problems get a bonus of `0.15 * streak`, capped at
   0.50.
5. Problems similar to the last 3 given this episode lose `0.4 * similarity`, which spreads selection
   across concept domains.
6. A difficulty floor removes problems that are too easy, applied only when the student has no
   consecutive failures and did not just give up.
7. A difficulty ceiling removes problems that are too hard. It widens in 0.05 steps until at least 20
   candidates remain, up to `mean_knowledge + 0.20`.
8. A softmax at temperature 0.2 samples the final choice.

## Reward

Per step:

| Condition | Reward |
| --- | ---: |
| Correct graded answer | `+1.5` |
| Wrong graded answer | `-1.0` |
| Ungraded success | `+1.0` |
| Ungraded failure | `-0.5` |
| Knowledge gain on active concepts | `+35 * delta` |
| Needs more training | `-0.3` |
| Mistake penalty | `-0.2 * penalty` |
| Motivation above 0.7 | `+0.1` |
| Motivation above 0.5 | `+0.05` |
| Motivation below 0.40 | `-1.0` |
| Gave up | `-40` |
| 2 or more consecutive failures | `-0.5` to `-1.5` |
| Difficulty within the target band | `+0.8` |
| Too easy by more than 0.10 | `-0.3` |
| Too hard by more than 0.10 | `-1.0` |
| Help requested below 0.4 confidence | `+0.5` |
| Help requested after a failure | `+0.2` |

The target difficulty band is `[mean_knowledge - 0.05, mean_knowledge + 0.15]`.

At the end of an episode:

| Condition | Reward |
| --- | ---: |
| Learning gain | `300 * max(0, NLG)` |
| Final mastery at or above 0.70 | `+50` |
| Completed all steps with real gain | up to `+8` |
| Motivation collapsed without giving up | `-20` |

`NLG` is `(final - initial) / (1 - initial)`, clipped to `[-1, 1]`, computed over only the concepts
practiced during that episode.

## Episode End

An episode truncates at `max_steps`, which defaults to 12.

It terminates early when motivation drops to 0.35 or below, or when fatigue reaches 0.85 with at
least two consecutive failures.

Giving up does not end the episode. It carries a large penalty and shapes later problem selection,
but the session continues.

## Problem Bank

```python
from rl_agent.problem_bank import ProblemBank

bank = ProblemBank.from_api("http://localhost:8000")
bank = ProblemBank.from_api("http://localhost:8000", min_difficulty=0.3, max_difficulty=0.7)
bank = ProblemBank.from_api_balanced("http://localhost:8000", target_count=300)
```

`from_api` pages through the API and accepts `max_problems`, `page_size`, `min_difficulty`,
`max_difficulty`, `concepts`, and `sub_concepts`.

`from_api_balanced` samples 30 percent easy at 0.55 or below, 40 percent medium between 0.55 and
0.72, and 30 percent hard above 0.72.

Useful members: `bank.problems`, `bank.concept_dim`, `bank.action_dim`, `bank.difficulty_array`,
`bank.action_matrix_norm`, `bank.concept_matrix_norm`, and `bank.get_problem_dict(index)`.

## Evaluation

```python
from rl_agent.evaluation_framework import EvaluationConfig, evaluate_policy

cfg = EvaluationConfig(episodes=100, compare_baselines=True, baseline_episodes=50)
summary = evaluate_policy(env, model, cfg)
summary.print_report()
```

`EvaluationConfig` fields:

| Field | Default | Purpose |
| --- | --- | --- |
| `episodes` | `50` | Episodes for the evaluated policy |
| `max_steps` | `None` | Override the environment episode length |
| `mastery_threshold` | `0.70` | Knowledge level counted as mastered |
| `window_splits` | `2` | Time windows for early versus late comparison |
| `compare_baselines` | `True` | Also run the baselines |
| `baseline_episodes` | `30` | Episodes per baseline |
| `capture_concept_vectors` | `True` | Keep full knowledge vectors for per-concept analysis |

Baselines: `random`, `fixed_medium`, `zpd_oracle`, `ascending_difficulty`, `lowest_mastery_first`.

The returned summary covers learning gain, mastery rate, give-up rate, per-student-type results,
knowledge quartiles, per-concept-domain gains, and baseline comparisons. Call `summary.to_dict()` for
a JSON-ready version.

## Training

Use the training script rather than driving PPO by hand:

```bash
python scripts/train_and_evaluate.py --api-base-url http://localhost:8000 --timesteps 100000
```
