"""
Problem bank loader for RL training.
Uses the RL-Math API as the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import numpy as np


@dataclass
class ProblemRecord:
    id: int
    question: str
    solution: str
    answer: Optional[str]
    concepts: List[str]
    sub_concepts: List[str]
    concept_vector: np.ndarray
    difficulty_score: float
    llm_reasoning: str
    estimated_time_minutes: int   # need to revisit this
    prerequisite_concepts: List[str] # same as sub_concepts for now


class ProblemBank:
    # 29 meaningful metadata dims  -  one flag + one continuous strength per domain.
    # Layout (see _encode_action for details):
    #   [0-4]   scope:        difficulty, n_concepts, n_sub_concepts, n_prereqs, est_time
    #   [5-7]   complexity:   question_len, solution_len, solution_steps (<<count)
    #   [8-16]  domain flags: arithmetic, fractions/dec, pct/ratio, algebra, geometry,
    #                         number_theory, statistics, time/money, problem_solving
    #                         (9 flags  -  one per concept vector group, complete coverage)
    #   [17-19] vector props: concept_sparsity, dominant_strength, answer_magnitude
    #   [20-28] group sums:   arithmetic, fractions/dec, pct/ratio, algebra, geometry,
    #                         number_theory, statistics, time/money, problem_solving
    #                         (continuous version of the 9 flags above)
    ACTION_META_DIM = 29

    def __init__(self, problems: List[ProblemRecord], action_meta_dim: int = ACTION_META_DIM):
        self.problems = problems
        self.action_meta_dim = action_meta_dim
        self.concept_dim = len(problems[0].concept_vector) if problems else 0
        self.action_dim = self.concept_dim + self.action_meta_dim
        self._action_matrix = self._build_action_matrix()
        self._action_matrix_norm = self._normalize_rows(self._action_matrix)
        # Pre-computed difficulty array for fast ceiling masking during training.
        # Shape (N,), dtype float32  -  same order as self.problems.
        self._difficulty_array = np.array(
            [p.difficulty_score for p in problems], dtype=np.float32
        ) if problems else np.empty(0, dtype=np.float32)
        # Pre-computed concept-only matrix (N, concept_dim), L2-normalised per row.
        # Used by the env's diversity penalty: cosine similarity between candidate
        # problems and the running average of recently-given problems' concept
        # vectors.  Built fresh (not sliced from _action_matrix_norm) because that
        # matrix is normalised over the full 87-dim action vector  -  its first 58
        # columns are NOT unit vectors.
        if problems:
            cm = np.vstack(
                [p.concept_vector.astype(np.float32) for p in problems]
            )
            self._concept_matrix_norm = self._normalize_rows(cm)
        else:
            self._concept_matrix_norm = np.zeros((0, self.concept_dim), dtype=np.float32)

    @property
    def action_matrix(self) -> np.ndarray:
        return self._action_matrix

    @property
    def action_matrix_norm(self) -> np.ndarray:
        return self._action_matrix_norm

    @property
    def difficulty_array(self) -> np.ndarray:
        """Shape (N,) float32 array of difficulty_score for every problem in order."""
        return self._difficulty_array

    @property
    def concept_matrix_norm(self) -> np.ndarray:
        """Shape (N, concept_dim) float32  -  L2-normalised concept vectors only.

        Used by the env's per-episode diversity penalty.  Each row is a unit
        vector so dot-products between this matrix and a unit query vector
        yield cosine similarity directly.
        """
        return self._concept_matrix_norm

    def indices_at_most(self, max_difficulty: float) -> np.ndarray:
        """Return integer indices of all problems with difficulty_score <= max_difficulty.

        Used by the tutoring env to hard-constrain problem selection after failures.
        Falls back to the N easiest problems (min 10) if the ceiling is too tight.
        """
        mask = self._difficulty_array <= max_difficulty
        indices = np.where(mask)[0]
        if len(indices) >= 10:
            return indices
        # Ceiling too tight  -  return the 10 easiest problems as a safe fallback
        return np.argsort(self._difficulty_array)[:10]

    def _build_action_matrix(self) -> np.ndarray:
        if not self.problems:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        actions = [self._encode_action(p) for p in self.problems]
        return np.vstack(actions).astype(np.float32)

    def _encode_action(self, problem: ProblemRecord) -> np.ndarray:
        cv = problem.concept_vector.astype(np.float32)  # 58-dim L2-normalised
        c_set = {c.lower() for c in problem.concepts}

        # --- Group 1: Scope (5 dims) ---
        meta: List[float] = [
            float(problem.difficulty_score),                          # 0
            float(len(problem.concepts)) / 10.0,                      # 1
            float(len(problem.sub_concepts)) / 20.0,                  # 2
            float(len(problem.prerequisite_concepts)) / 10.0,         # 3
            float(problem.estimated_time_minutes) / 30.0,             # 4
        ]

        # --- Group 2: Text complexity (3 dims) ---
        # solution uses "<<expr=result>>" markers for each arithmetic operation
        solution_steps = problem.solution.count("<<")
        meta += [
            min(1.0, len(problem.question) / 500.0),                  # 5
            min(1.0, len(problem.solution) / 2000.0),                  # 6
            min(1.0, float(solution_steps) / 10.0),                   # 7
        ]

        # --- Group 3: Domain-type flags (9 dims) ---
        # One binary flag per concept vector group  -  mirrors Group 5 below.
        # Derived from the concepts list (high-level names from CONCEPT_TO_SUBCONCEPTS).
        meta += [
            # 8  Arithmetic: addition, subtraction, multiplication, division, order_of_operations
            1.0 if c_set & {"addition", "subtraction", "multiplication",
                            "division", "order_of_operations"} else 0.0,
            # 9  Fractions/Decimals: fractions, decimals, fraction_to_decimal_conversion
            1.0 if c_set & {"fractions", "decimals",
                            "fraction_to_decimal_conversion"} else 0.0,
            # 10 Percentages/Ratios: percentages, ratios, proportions, rates
            1.0 if c_set & {"percentages", "ratios",
                            "proportions", "rates"} else 0.0,
            # 11 Algebra: variables, algebraic_expressions, solving_equations
            1.0 if c_set & {"variables", "algebraic_expressions",
                            "solving_equations"} else 0.0,
            # 12 Geometry/Measurement: area, perimeter, volume, angles, shapes,
            #                          measurement, unit_conversion
            1.0 if c_set & {"area", "perimeter", "volume", "angles",
                            "shapes", "measurement",
                            "unit_conversion"} else 0.0,
            # 13 Number Theory: factors, multiples, primes, divisibility
            1.0 if c_set & {"factors", "multiples", "primes",
                            "divisibility"} else 0.0,
            # 14 Statistics/Probability: mean_average, median, mode, range,
            #                            probability, counting, data_interpretation
            1.0 if c_set & {"mean_average", "median", "mode", "range",
                            "probability", "counting",
                            "data_interpretation"} else 0.0,
            # 15 Time/Money: time_calculations, elapsed_time, money_calculations, currency
            1.0 if c_set & {"time_calculations", "elapsed_time",
                            "money_calculations", "currency"} else 0.0,
            # 16 Problem-Solving: word_problem_comprehension, multi_step_problem_solving,
            #                     logical_reasoning, pattern_recognition, estimation
            1.0 if c_set & {"word_problem_comprehension", "multi_step_problem_solving",
                            "logical_reasoning", "pattern_recognition",
                            "estimation"} else 0.0,
        ]

        # --- Group 4: Concept-vector properties (3 dims) ---
        nonzero = int(np.count_nonzero(cv))
        meta += [
            float(nonzero) / 58.0,                                    # 17 sparsity
            float(np.max(cv)) if nonzero > 0 else 0.0,               # 18 dominant strength
            self._answer_magnitude(problem.answer),                    # 19 answer scale
        ]

        # --- Group 5: Per-domain group strengths (9 dims, per API vector layout) ---
        # Continuous version of Group 3 flags  -  sum of vector weights in each domain.
        # Clipped to 1.0 because group sums of a normalised vector can exceed 1.
        meta += [
            min(1.0, float(np.sum(cv[0:10]))),   # 20 Arithmetic (dims 0-9)
            min(1.0, float(np.sum(cv[10:18]))),  # 21 Fractions/Decimals (10-17)
            min(1.0, float(np.sum(cv[18:24]))),  # 22 Percentages/Ratios (18-23)
            min(1.0, float(np.sum(cv[24:30]))),  # 23 Algebra (24-29)
            min(1.0, float(np.sum(cv[30:40]))),  # 24 Geometry/Measurement (30-39)
            min(1.0, float(np.sum(cv[40:44]))),  # 25 Number Theory (40-43)
            min(1.0, float(np.sum(cv[44:49]))),  # 26 Statistics/Probability (44-48)
            min(1.0, float(np.sum(cv[49:51]))),  # 27 Time/Money (49-50)
            min(1.0, float(np.sum(cv[51:58]))),  # 28 Problem-Solving Skills (51-57)
        ]

        assert len(meta) == self.ACTION_META_DIM, f"Expected {self.ACTION_META_DIM} meta dims, got {len(meta)}"
        return np.concatenate([cv, np.array(meta, dtype=np.float32)])

    @staticmethod
    def _answer_magnitude(answer: Optional[str]) -> float:
        """Encode numeric answer as log-scale magnitude, normalised to [0, 1].

        Covers answers in range 1–1_000_000 (log10 range 0–6).
        Returns 0.5 for missing or non-numeric answers.
        """
        if not answer:
            return 0.5
        try:
            val = abs(float(str(answer).replace(",", "").strip()))
            if val <= 0:
                return 0.0
            return min(1.0, math.log10(val + 1) / 6.0)
        except (ValueError, AttributeError):
            return 0.5

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    @classmethod
    def from_api(
        cls,
        base_url: str = "http://localhost:8000",
        max_problems: Optional[int] = None,
        page_size: int = 500,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
        concepts: Optional[List[str]] = None,
        sub_concepts: Optional[List[str]] = None,
    ) -> "ProblemBank":
        base_url = base_url.rstrip("/")
        page_size = min(max(1, page_size), 500)

        problems: List[ProblemRecord] = []
        page = 1
        expected_dim: Optional[int] = None

        while True:
            remaining = None
            if max_problems is not None:
                remaining = max_problems - len(problems)
                if remaining <= 0:
                    break

            effective_page_size = page_size
            if remaining is not None:
                effective_page_size = min(page_size, remaining)

            params: Dict[str, Any] = {
                "page": page,
                "page_size": effective_page_size,
            }
            if min_difficulty is not None:
                params["min_difficulty"] = min_difficulty
            if max_difficulty is not None:
                params["max_difficulty"] = max_difficulty
            if concepts:
                params["concepts"] = ",".join(concepts)
            if sub_concepts:
                params["sub_concepts"] = ",".join(sub_concepts)

            query = urlencode(params)
            url = f"{base_url}/api/v1/problems?{query}"

            try:
                with urlopen(url, timeout=15) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError) as exc:
                raise RuntimeError(f"API request failed: {url}") from exc

            items = payload.get("items", [])
            if not items:
                break

            for item in items:
                question = str(item.get("question", "")).strip()
                solution = str(item.get("solution", "")).strip()
                if not question:
                    continue

                concept_vector = item.get("concept_vector", [])
                if not concept_vector:
                    continue

                concept_vec = np.array(concept_vector, dtype=np.float32)
                if expected_dim is None:
                    expected_dim = concept_vec.shape[0]
                elif concept_vec.shape[0] != expected_dim:
                    continue
                difficulty = float(item.get("difficulty_score", 0.5))
                estimated_time_minutes = cls._estimate_time_minutes(difficulty)

                concepts_list = item.get("concepts", []) or []
                sub_concepts_list = item.get("sub_concepts", []) or []
                prereqs = item.get("prerequisite_concepts", None)
                if prereqs is None:
                    prereqs = sub_concepts_list

                problems.append(
                    ProblemRecord(
                        id=int(item.get("id", len(problems))),
                        question=question,
                        solution=solution,
                        answer=item.get("answer"),
                        concepts=concepts_list,
                        sub_concepts=sub_concepts_list,
                        concept_vector=concept_vec,
                        difficulty_score=difficulty,
                        llm_reasoning=str(item.get("llm_reasoning", "")).strip(),
                        estimated_time_minutes=estimated_time_minutes,
                        prerequisite_concepts=prereqs,
                    )
                )

                if max_problems is not None and len(problems) >= max_problems:
                    break

            if max_problems is not None and len(problems) >= max_problems:
                break

            total_pages = payload.get("pages")
            if total_pages is not None and page >= total_pages:
                break
            page += 1

        if not problems:
            raise ValueError("No problems loaded from API")

        return cls(problems)

    @classmethod
    def from_api_balanced(
        cls,
        base_url: str = "http://localhost:8000",
        target_count: int = 300,
        page_size: int = 500,
        easy_ratio: float = 0.30,
        medium_ratio: float = 0.40,
        hard_ratio: float = 0.30,
    ) -> "ProblemBank":
        """Load problems with balanced difficulty distribution.

        Fetches all available problems then stratified-samples to get a mix
        of easy (<=0.55), medium (0.55-0.72), and hard (>0.72) problems.
        Thresholds are tuned for the full annotated GSM8K training dataset,
        whose difficulty scores span 0.4108-0.8905 with mean 0.7163.
        """
        all_bank = cls.from_api(base_url=base_url, page_size=page_size)

        easy = [p for p in all_bank.problems if p.difficulty_score <= 0.55]
        medium = [p for p in all_bank.problems if 0.55 < p.difficulty_score <= 0.72]
        hard = [p for p in all_bank.problems if p.difficulty_score > 0.72]

        rng = np.random.default_rng(42)

        def _sample(bucket: list, n: int) -> list:
            if len(bucket) <= n:
                return list(bucket)
            indices = rng.choice(len(bucket), size=n, replace=False)
            return [bucket[i] for i in indices]

        n_easy = max(1, int(target_count * easy_ratio))
        n_medium = max(1, int(target_count * medium_ratio))
        n_hard = max(1, int(target_count * hard_ratio))

        sampled = (
            _sample(easy, n_easy)
            + _sample(medium, n_medium)
            + _sample(hard, n_hard)
        )
        rng.shuffle(sampled)

        if not sampled:
            raise ValueError("No problems after balanced sampling")

        return cls(sampled)

    @staticmethod
    def _estimate_time_minutes(difficulty_score: float) -> int:
        base = 2.0
        scaled = base + (difficulty_score * 6.0)
        return max(1, int(round(scaled)))

    def get_problem_dict(self, index: int) -> Dict[str, Any]:
        problem = self.problems[index]
        return {
            "id": problem.id,
            "question": problem.question,
            "solution": problem.solution,
            "answer": problem.answer,
            "concepts": problem.concepts,
            "sub_concepts": problem.sub_concepts,
            "concept_vector": problem.concept_vector,
            "difficulty_score": problem.difficulty_score,
            "llm_reasoning": problem.llm_reasoning,
            "estimated_time_minutes": problem.estimated_time_minutes,
            "prerequisite_concepts": problem.prerequisite_concepts,
        }
