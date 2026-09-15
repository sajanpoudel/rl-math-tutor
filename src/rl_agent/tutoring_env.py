"""
Gym-compatible tutoring environment using the Stage 5 problem bank.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces
    _GYMNASIUM = False

from simulation.student_model import CognitiveStudentModel, StudentPopulation
from simulation.llm_answer_generator import build_dual_agents_from_env
from rl_agent.problem_bank import ProblemBank


class SimulatedTutoringEnv(gym.Env):
    """Tutoring environment with continuous action selection via vector matching."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        max_problems: Optional[int] = None,
        max_steps: int = 12,
        student_profile: str = "population",
        num_students: int = 200,
        seed: Optional[int] = None,
        page_size: int = 500,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
        concepts: Optional[List[str]] = None,
        sub_concepts: Optional[List[str]] = None,
        balanced: bool = False,
    ):
        if balanced:
            self.problem_bank = ProblemBank.from_api_balanced(
                base_url=api_base_url,
                target_count=max_problems or 300,
                page_size=page_size,
            )
        else:
            self.problem_bank = ProblemBank.from_api(
                base_url=api_base_url,
                max_problems=max_problems,
                page_size=page_size,
                min_difficulty=min_difficulty,
                max_difficulty=max_difficulty,
                concepts=concepts,
                sub_concepts=sub_concepts,
            )
        self.concept_dim = self.problem_bank.concept_dim
        self.context_dim = 16  # exactly 16 real context features, no zero padding
        self.action_meta_dim = self.problem_bank.action_meta_dim
        self._student_types = [
            "average", "analytical", "intuitive",
            "fast_learner", "struggling_learner",
            "anxious_learner", "confident_learner",
        ]
        self.student_type_dim = len(self._student_types)  # 7
        self.state_dim = self.concept_dim + self.context_dim + self.student_type_dim
        self.action_dim = self.concept_dim + self.action_meta_dim

        self.max_steps = max_steps
        self.student_profile = student_profile
        self.student_population = StudentPopulation(
            num_students=num_students,
            concept_dim=self.concept_dim,
        )
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self._current_student: Optional[CognitiveStudentModel] = None
        self._step_count = 0
        # Remediation state: concepts the student failed on in the last step
        self._pending_practice_targets: List[str] = []
        self._pending_hint: str = ""
        # Weight applied to the remediation concept-boost in problem selection.
        # Scales with failure severity: 0.0 (no failure) → 0.30 (wrong) → 0.60 (give-up).
        self._remediation_weight: float = 0.0
        # Tracks knowledge at episode start for terminal NLG reward computation
        self._episode_initial_knowledge: float = 0.0
        # Track which concept indices were active during the episode for
        # accurate terminal NLG computation over practiced concepts only.
        self._episode_active_concept_indices: set = set()
        # Snapshot of full knowledge state at episode start (per-concept).
        # Used with _episode_active_concept_indices for active-concept NLG.
        self._episode_initial_knowledge_state: np.ndarray = np.zeros(1)
        # Buffer of recent problems' concept vectors (current episode only).
        # Drives the diversity penalty in problem selection: candidates whose
        # concepts heavily overlap with this buffer are downweighted.  Without
        # this, cosine retrieval on the 87-dim action vector concentrates on
        # word-problem dims (51, 53)  -  every checkpoint shows ~30× higher
        # gain on problem-solving than other domains, capping mastery rate.
        # Cleared on episode reset.
        self._recent_problem_concept_vecs: List[np.ndarray] = []
        self._diversity_buffer_size: int = 3
        # Strength of the diversity penalty term (subtracted from cosine
        # scores).  0.4 is empirically calibrated: cosine scores are in
        # [-1, 1], so a 0.4 penalty on a perfectly-overlapping problem
        # cuts its score by 40%  -  strong enough to nudge selection toward
        # different concepts when the action vector is ambiguous, but not
        # so strong that the policy can't override it for genuinely
        # remediation-targeted problems.
        self._diversity_penalty_strength: float = 0.4

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._step_count = 0
        self._pending_practice_targets = []
        self._pending_hint = ""
        self._remediation_weight = 0.0
        # Clear the diversity buffer so each new episode starts with a clean
        # slate  -  prior episode's concept choices shouldn't influence the
        # first problem of the next student.
        self._recent_problem_concept_vecs = []

        # Allow caller to inject a specific student for reproducible probe evaluation.
        # Used by CheckpointEvalCallback to run the same fixed student set at every
        # checkpoint so NLG comparisons are meaningful across training steps.
        probe_student = (options or {}).get("probe_student")
        if probe_student is not None:
            self._current_student = probe_student
        else:
            self._current_student = self._create_student()
        self._current_student.reset_session()
        # Snapshot initial knowledge for terminal NLG computation at episode end
        self._episode_initial_knowledge = float(np.mean(self._current_student.knowledge_state))
        self._episode_active_concept_indices = set()
        self._episode_initial_knowledge_state = self._current_student.knowledge_state.copy()
        obs = self._get_observation()
        info = {
            "student_id": self._current_student.student_id,
            "student_type": self._current_student.student_type,
        }

        if _GYMNASIUM:
            return obs, info
        return obs

    def step(self, action: np.ndarray):
        if self._current_student is None:
            raise RuntimeError("Environment must be reset before stepping")

        self._step_count += 1
        # Clip and normalise the incoming action.
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Action dim {action.shape[0]} does not match {self.action_dim}")

        action = np.clip(action, 0.0, 1.0)
        norm = np.linalg.norm(action)
        if norm > 0:
            action = action / norm

        # Capture state before the step for learning/engagement diagnostics.
        _state_before = self._current_student.get_current_state()
        _knowledge_before = float(np.mean(_state_before["knowledge_state"]))
        _motivation_before = float(_state_before.get("motivation_level", 0.0))

        # Forward any remediation hint from the previous failed step so the
        # student model can surface it as prior guidance in the epistemic spec.
        if self._pending_hint and self._current_student is not None:
            self._current_student._pending_remediation_hint = self._pending_hint

        logger.info(
            "┌── STEP %d | student=%s | K_mean=%.4f | motivation=%.3f | fatigue=%.3f",
            self._step_count,
            self._current_student.student_type,
            _knowledge_before,
            _motivation_before,
            getattr(self._current_student, "fatigue_level", 0.0),
        )

        problem = self._select_problem_from_action(action)
        # Append the chosen problem's concept vector to the diversity buffer
        # (FIFO of last _diversity_buffer_size problems).  Used by the next
        # call to _select_problem_from_action to penalise candidates whose
        # concepts overlap heavily with what was just asked.  Reset at
        # episode boundary so prior students don't leak.
        _problem_cv = np.asarray(
            problem.get("concept_vector", []), dtype=np.float32
        )
        if _problem_cv.size == self.problem_bank.concept_dim:
            self._recent_problem_concept_vecs.append(_problem_cv)
            if len(self._recent_problem_concept_vecs) > self._diversity_buffer_size:
                self._recent_problem_concept_vecs.pop(0)
        logger.info(
            "│   PROBLEM id=%s | diff=%.3f | concepts=%s",
            problem.get("id"),
            problem.get("difficulty_score", 0.0),
            problem.get("concepts", []),
        )
        logger.debug("│   QUESTION:  %s", problem.get("question", ""))
        logger.debug("│   SOLUTION:  %s", problem.get("solution", ""))
        logger.debug("│   ANSWER:    %s", problem.get("answer", ""))
        logger.debug("│   SUB_CONCEPTS: %s", problem.get("sub_concepts", []))
        logger.debug("│   PREREQS:      %s", problem.get("prerequisite_concepts", []))

        result = self._current_student.attempt_problem(problem)

        _state_after = self._current_student.get_current_state()
        # Compute knowledge delta over the problem's ACTIVE concepts only.
        # Using the full 58-concept mean dilutes the signal by ~10x because most
        # concepts are inactive per problem. This was a root cause of the agent
        # learning to ignore knowledge gain as a reward signal.
        _concept_vector = np.array(problem.get("concept_vector", []), dtype=np.float32)
        _active_idx = np.where(_concept_vector > 0.0)[0]
        if len(_active_idx) > 0:
            _knowledge_delta = (
                float(np.mean(_state_after["knowledge_state"][_active_idx]))
                - float(np.mean(_state_before["knowledge_state"][_active_idx]))
            )
        else:
            _knowledge_delta = float(np.mean(_state_after["knowledge_state"])) - _knowledge_before
        # Accumulate concept indices practiced this episode for active-concept NLG.
        self._episode_active_concept_indices.update(_active_idx.tolist())
        _motivation_after = float(_state_after.get("motivation_level", 0.0))
        _motivation_delta = _motivation_after - _motivation_before

        reward = self._calculate_reward(result, self._current_student, problem, _knowledge_delta)

        logger.info(
            "│   RESULT  success=%s | correct=%s | gave_up=%s | confidence=%.3f | answer='%s'",
            result.get("success"),
            result.get("answer_correct"),
            result.get("gave_up"),
            result.get("confidence", 0.0),
            str(result.get("final_answer", ""))[:60],
        )
        logger.info(
            "│   ANSWER  '%s'",
            str(result.get("answer_text", ""))[:200],
        )
        logger.info(
            "└── REWARD  %.3f | ΔK=%+.5f | Δmotivation=%+.3f | llm_failed=%s",
            reward,
            _knowledge_delta,
            _motivation_delta,
            result.get("llm_failed", False),
        )

        # Update remediation state for the next step.
        # Fires on any failure  -  wrong answer OR give-up.
        # Give-up is the strongest signal: boost weight is highest and targets
        # are set regardless of needs_more_training so the next step always
        # gets a difficulty ceiling and concept bias after a disengage event.
        if not result["success"]:
            self._pending_practice_targets = list(result.get("practice_targets", []))
            self._pending_hint = str(result.get("hint_text", ""))
            if result["gave_up"]:
                self._remediation_weight = 0.60  # maximum: student fully disengaged
            else:
                self._remediation_weight = 0.30  # moderate: wrong answer
        else:
            self._pending_practice_targets = []
            self._pending_hint = ""
            self._remediation_weight = 0.0

        # Implicit encouragement: when the student is struggling (motivation < 0.40
        # AND consecutive failures ≥ 2), the tutor applies a scaffolding nudge  -
        # a small motivation recovery representing verbal encouragement or a brief
        # reframe.  Modelled as a state update here rather than a separate action
        # to avoid changing the action space while still capturing the effect.
        if (self._current_student.motivation_level < 0.40
                and self._current_student.consecutive_failures >= 2):
            self._current_student.motivation_level = min(
                1.0, self._current_student.motivation_level + 0.06
            )
            logger.info("│   ENCOURAGEMENT  motivation +0.06")

        # Give-up is no longer a terminal condition.  The −40 penalty in the
        # reward signal already penalises it; terminating removes the agent's
        # opportunity to recover the session with an easier problem.
        #
        # Two natural termination conditions  -  both represent genuine session end:
        #
        # 1. Motivation drain: motivation ≤ 0.35  -  student has fully disengaged.
        #    The encouragement mechanic (+0.06 at motivation < 0.40) can keep a
        #    struggling student alive temporarily, but once they hit 0.35 the
        #    session truly cannot continue.
        #
        # 2. Cognitive exhaustion: fatigue ≥ 0.85 AND 2+ consecutive failures.
        #    At fatigue=0.85, effective learning rate ≈ learning_rate × 0.15 ≈ 0.01
        #     -  essentially zero.  Without this condition, high-motivation students
        #    (average, analytical, confident) can be forced through all max_steps
        #    in a fully-fatigued state where no learning occurs.  This was the
        #    core reason max_steps=20 produced ~50% noise transitions in the
        #    rollout buffer for low-attention-span student types.
        _fatigue = getattr(self._current_student, "fatigue_level", 0.0)
        _consec_fail = getattr(self._current_student, "consecutive_failures", 0)
        terminated = (
            self._current_student.motivation_level <= 0.35
            or (_fatigue >= 0.85 and _consec_fail >= 2)
        )
        truncated = self._step_count >= self.max_steps

        # Motivation-drain terminal penalty: fires when the episode ends because
        # the student's motivation collapsed (not because they explicitly gave up,
        # which already carries its own -40 penalty in _calculate_reward).
        # Without this, motivation drain was a free escape hatch  -  the episode
        # ended with no terminal cost, which the agent exploited.
        if terminated and not result["gave_up"] and self._current_student.motivation_level <= 0.35:
            reward -= 20.0
            logger.info("│   MOTIVATION-DRAIN TERMINAL  -20.0")

        # Terminal NLG reward: the dominant pedagogical signal.
        # Grounded in Clement et al. (2015) ITS methodology: NLG = (post - pre) / (1 - pre).
        # Multiplied by 80 so that a typical productive session (NLG ≈ 0.05–0.15)
        # produces a terminal bonus of +4 to +12, which outweighs per-step noise.
        # Session completion bonus (+8) rewards running all max_steps, directly
        # counteracting the early-termination exploitation seen in steps 6000–7000.
        if terminated or truncated:
            # Terminal NLG: compute over concepts actually practiced this episode.
            # Using full 58-concept mean dilutes the signal ~10× because most
            # concepts are inactive per problem  -  the same bug that was fixed for
            # per-step knowledge_delta.
            if self._episode_active_concept_indices:
                _active_idx_ep = np.array(
                    sorted(self._episode_active_concept_indices), dtype=int
                )
                _final_k = float(
                    np.mean(self._current_student.knowledge_state[_active_idx_ep])
                )
                _initial_k_ep = float(
                    np.mean(self._episode_initial_knowledge_state[_active_idx_ep])
                )
            else:
                _final_k = float(np.mean(self._current_student.knowledge_state))
                _initial_k_ep = self._episode_initial_knowledge

            _nlg = (_final_k - _initial_k_ep) / max(1.0 - _initial_k_ep, 1e-3)
            _nlg = float(np.clip(_nlg, -1.0, 1.0))
            _nlg_reward = 300.0 * max(0.0, _nlg)
            if _final_k >= 0.7:
                _nlg_reward += 50.0   # mastery bonus
            # Session completion bonus: only awarded when genuine learning occurred.
            # Unconditional +8 was exploitable  -  agent could collect it with 0 NLG.
            if truncated and not terminated and _nlg > 0.002:
                completion_scale = min(1.0, (_nlg - 0.002) / 0.018)
                _nlg_reward += 8.0 * completion_scale
            reward += _nlg_reward
            logger.info(
                "│   TERMINAL  NLG=%.4f  reward=+%.2f  "
                "(initial_K=%.4f  final_K=%.4f  mastery=%s  completed=%s  "
                "active_concepts=%d)",
                _nlg, _nlg_reward,
                _initial_k_ep, _final_k,
                _final_k >= 0.7,
                truncated and not terminated,
                len(self._episode_active_concept_indices),
            )

        obs = self._get_observation()
        info = {
            "problem_id": problem["id"],
            "difficulty": problem["difficulty_score"],
            "success": result["success"],
            "gave_up": result["gave_up"],
            "final_answer": result.get("final_answer"),
            "answer_correct": result.get("answer_correct"),
            "answer_was_graded": result.get("answer_was_graded"),
            "knowledge_delta": _knowledge_delta,
            "motivation_delta": _motivation_delta,
            # Richer signals for evaluation and reward shaping
            "confidence": result.get("confidence", 0.5),
            "motivation": _motivation_after,
            "student_type": self._current_student.student_type,
            "needs_more_training": result.get("needs_more_training", False),
        }

        if _GYMNASIUM:
            return obs, reward, terminated, truncated, info
        done = terminated or truncated
        return obs, reward, done, info

    def _create_student(self) -> CognitiveStudentModel:
        if self.student_profile == "dull":
            decision_agent, solver_agent = build_dual_agents_from_env("average")
            return CognitiveStudentModel(
                student_id="dull_student",
                student_type="average",
                initial_knowledge_level=0.3,
                learning_rate=0.1,
                attention_span=10,
                motivation_level=0.6,
                concept_dim=self.concept_dim,
                decision_agent=decision_agent,
                solver_agent=solver_agent,
            )
        return self.student_population.sample_student()

    def _get_difficulty_ceiling(self) -> Optional[float]:
        """Compute the maximum allowed problem difficulty for this step.

        Returns None when no ceiling applies (clean slate  -  agent has full
        control).  When the student has been struggling, the ceiling is
        clamped to their current knowledge level plus a small slack so the
        next problem stays inside their Zone of Proximal Development.

        Progressive tightening schedule (raw difficulty_score scale):
          0 failures / clean   → None   (no constraint)
          1 consecutive failure → K_mean + 0.12  (gentle nudge toward ZPD)
          2 consecutive failures→ K_mean + 0.08  (moderate ZPD ceiling)
          3+ failures           → K_mean + 0.05  (tight ZPD ceiling)
          last step was give-up → K_mean + 0.03  (near current level  -  re-engage
                                                   with something achievable)

        The difficulty_score range in the bank is [0.41, 0.89].  K_mean for
        typical students is [0.52, 0.78].  So K_mean + 0.05 ≈ 0.57–0.83,
        which covers most of the bank and still allows challenge.
        """
        if self._current_student is None:
            return None

        consec = int(self._current_student.consecutive_failures)
        last_gave_up = bool(self._current_student.last_gave_up)
        k_mean = float(np.mean(self._current_student.knowledge_state))

        if last_gave_up:
            return k_mean + 0.03
        if consec >= 3:
            return k_mean + 0.05
        if consec >= 2:
            return k_mean + 0.08
        if consec >= 1:
            return k_mean + 0.12
        # Already-mastered students: cap at consolidation range even with no
        # failures.  Prevents the agent assigning maximum-difficulty problems
        # to fast_learners/confident_learners who already cleared the 0.70
        # threshold  -  they need consolidation, not extreme over-challenge.
        if k_mean >= 0.70:
            return k_mean + 0.10
        # Weak students (struggling territory): always cap at K+0.05 even
        # with zero consecutive failures.  Without this, the first 1-3
        # problems are unconstrained and the agent virtually always picks
        # medium/hard problems (dataset mean diff = 0.7163 and median diff
        # = 0.7230, well above
        # struggling K = 0.50-0.57).  By the time the failure-triggered
        # ceiling tightens, motivation has already drained.  This shifts
        # struggling students into the easy/medium-easy range from step 1.
        if k_mean < 0.60:
            return k_mean + 0.05
        return None  # no ceiling  -  normal operation

    def _get_difficulty_floor(self) -> Optional[float]:
        """Compute the minimum allowed problem difficulty for this step.

        Returns None when no floor applies  -  the agent can choose downward
        freely.  A floor is set ONLY for high-knowledge students who are
        currently performing well (no consecutive failures, no recent
        give-up).  This prevents the agent from cheap-out behaviour where
        already-strong students get trivially-easy problems for guaranteed
        success-bonus reward.

        Without this floor, fast_learners (K=0.74) routinely received
        difficulty=0.71 problems  -  barely above their level  -  across every
        checkpoint.  With the floor, K≥0.65 students get problems in
        [K_mean−0.05, ceiling], pushing the upper end of the
        difficulty-differentiation distribution that was stuck at ~0.07.

        The floor is suspended whenever the student starts struggling
        (consec_failures≥1 or last_gave_up) so they can get easier
        problems for re-engagement regardless of starting K.
        """
        if self._current_student is None:
            return None

        consec = int(self._current_student.consecutive_failures)
        last_gave_up = bool(self._current_student.last_gave_up)
        if consec >= 1 or last_gave_up:
            return None

        k_mean = float(np.mean(self._current_student.knowledge_state))
        if k_mean >= 0.65:
            return k_mean - 0.05
        return None

    def _select_problem_from_action(self, action: np.ndarray) -> Dict[str, Any]:
        concept_dim = self.problem_bank.concept_dim

        # Concept-weighted similarity: concept dims carry the personalization
        # signal (which skills to target). We amplify them 2× relative to the
        # structural meta dims before computing cosine similarity.
        action_w = action.copy()
        action_w[:concept_dim] *= 2.0
        norm = np.linalg.norm(action_w)
        if norm > 0:
            action_w = action_w / norm

        scores = action_w @ self.problem_bank.action_matrix_norm.T

        # Remediation concept-bias: after a failure or give-up, steer toward
        # problems covering the weak concepts identified in the previous step.
        # Weight scales with severity: 0.30 (wrong answer) or 0.60 (give-up).
        if self._pending_practice_targets and self._remediation_weight > 0.0:
            boost = self._compute_remediation_boost(self._pending_practice_targets)
            scores = scores + self._remediation_weight * boost

        # Difficulty-downward bias after consecutive failures.  The hard
        # ceiling alone isn't enough  -  within a wide eligible pool, cosine
        # similarity still picks medium-range problems (concentration around
        # diff=0.65-0.72).  This adds a downward pressure on score for harder
        # problems within the eligible pool, so the softmax actively prefers
        # the easier end of the range, not just any problem under the ceiling.
        # Empirically the remediation_followup_success_rate was 0-7% across
        # checkpoints because the agent kept selecting same-level problems
        # after failure.  Bias strength scales with failure streak.
        consec = (
            self._current_student.consecutive_failures
            if self._current_student is not None
            else 0
        )
        if consec >= 2:
            diff_arr = self.problem_bank.difficulty_array  # (N,)
            diff_min, diff_max = 0.40, 0.90
            diff_norm = np.clip(
                (diff_arr - diff_min) / (diff_max - diff_min), 0.0, 1.0
            )
            bias_strength = min(0.5, 0.15 * consec)  # 0.30 (2), 0.45 (3), cap 0.50
            scores = scores + bias_strength * (1.0 - diff_norm)

        # Concept-diversity penalty: subtract cosine similarity to recently-
        # given problems' average concept vector.  Without this, the agent
        # converges on a narrow region of the 58-dim concept space (every
        # checkpoint shows ~30× higher gain on problem-solving dims 51,53
        # than on geometry/number-theory).  The penalty pushes selection
        # toward concepts that have NOT been activated in the last few
        # problems of the current episode, breaking the per-domain trap
        # so K_mean can move across more dimensions and mastery rate can
        # actually rise above 21%.  Buffer is per-episode (cleared on reset).
        if (
            self._recent_problem_concept_vecs
            and self.problem_bank.concept_matrix_norm.shape[0] > 0
        ):
            recent_avg = np.mean(self._recent_problem_concept_vecs, axis=0)
            rn = float(np.linalg.norm(recent_avg))
            if rn > 0:
                recent_avg = recent_avg / rn
                # Cosine similarity (N,): each row of concept_matrix_norm is
                # already L2-normalised, recent_avg is now unit length too.
                sim_to_recent = (
                    self.problem_bank.concept_matrix_norm @ recent_avg
                )
                scores = scores - self._diversity_penalty_strength * sim_to_recent

        # Hard difficulty floor for high-K students who are performing well:
        # without a floor, the agent learns to give already-strong students
        # easy problems for cheap success-bonus reward.  fast_learners at
        # K=0.74 routinely got diff=0.71 problems (barely above their K).
        # Floor only applies when the student has zero consecutive failures
        # AND did not just give up  -  when they're struggling, they need
        # easier problems regardless of starting K.  Mask is applied BEFORE
        # the ceiling so the ceiling's MIN_POOL widening still works on the
        # remaining eligible pool.
        floor = self._get_difficulty_floor()
        if floor is not None:
            diff_arr = self.problem_bank.difficulty_array
            below_floor = diff_arr < floor
            scores = np.where(below_floor, -np.inf, scores)

        # Hard difficulty ceiling: mask out any problem above the ceiling so
        # the softmax never selects it.  This is the primary fix for the
        # agent assigning problems too hard for struggling/failing students.
        # The ceiling is computed from the student's current knowledge level
        # and tightens progressively with each consecutive failure.
        ceiling = self._get_difficulty_ceiling()
        if ceiling is not None:
            valid_indices = self.problem_bank.indices_at_most(ceiling)
            # Minimum pool guard: if the ceiling is so tight that fewer than
            # MIN_POOL problems are eligible (e.g. struggling_learner with only
            # 27 easy problems), widen it in 0.05 steps until we have enough
            # options.  This prevents the extreme fallback (10 easiest) from
            # firing on every step for sparse difficulty regions.
            # Cap at k_mean + 0.20 so we never jump too far above ZPD.
            MIN_POOL = 20
            original_ceiling = ceiling
            k_mean = float(np.mean(self._current_student.knowledge_state)) if self._current_student else 0.6
            max_ceiling = k_mean + 0.20
            while len(valid_indices) < MIN_POOL and ceiling < max_ceiling:
                ceiling += 0.05
                valid_indices = self.problem_bank.indices_at_most(ceiling)
            # Set scores of out-of-range problems to −∞ so softmax ignores them.
            mask = np.full(len(scores), -np.inf, dtype=np.float32)
            mask[valid_indices] = scores[valid_indices]
            scores = mask
            logger.info(
                "│   DIFFICULTY CEILING  %.3f→%.3f | eligible=%d/%d | consec_fail=%d | gave_up=%s",
                original_ceiling,
                ceiling,
                len(valid_indices),
                len(scores),
                self._current_student.consecutive_failures if self._current_student else 0,
                self._current_student.last_gave_up if self._current_student else False,
            )

        # Softmax selection with temperature for exploration.
        # Increased from 0.1 (near-greedy) to 0.2 for more problem diversity.
        # At 0.1, repeated action vectors select the same 1-2 problems ~90% of
        # the time; at 0.2, the distribution is meaningfully broader while still
        # favouring higher-scoring problems.
        temperature = 0.2
        logits = scores / max(temperature, 1e-8)
        logits = logits - logits[np.isfinite(logits)].max()  # numerical stability over valid subset
        probs = np.exp(logits)
        probs = np.where(np.isfinite(logits), probs, 0.0)
        prob_sum = probs.sum()
        if prob_sum <= 0.0:
            # Extreme fallback: uniform over the 10 easiest problems
            fallback = np.argsort(self.problem_bank.difficulty_array)[:10]
            probs = np.zeros(len(scores))
            probs[fallback] = 1.0 / len(fallback)
        else:
            probs = probs / prob_sum
        selected_idx = int(self._rng.choice(len(probs), p=probs))
        return self.problem_bank.get_problem_dict(selected_idx)

    def _compute_remediation_boost(self, practice_targets: List[str]) -> np.ndarray:
        """Return a (N_problems,) score in [0,1] based on concept target overlap.

        Problems covering more of the student's identified weak concepts score
        higher.  Multiplied by self._remediation_weight (0.30 on wrong answer,
        0.60 on give-up) and added to cosine scores before softmax selection.
        Applied after the hard difficulty ceiling mask so the boost only acts
        within the already-filtered candidate set.
        """
        boost = np.zeros(len(self.problem_bank.problems), dtype=np.float32)
        concept_targets: set = set()
        sub_targets: set = set()
        vector_targets: set = set()

        for t in practice_targets:
            t = str(t).strip()
            if t.startswith("concept::"):
                name = t.split("::", 1)[1]
                if name.startswith("concept_idx_"):
                    try:
                        vector_targets.add(int(name.split("_")[-1]))
                    except ValueError:
                        pass
                else:
                    concept_targets.add(name.lower())
            elif t.startswith("sub_concept::"):
                sub_targets.add(t.split("::", 1)[1].lower())

        for i, prob in enumerate(self.problem_bank.problems):
            score = 0.0
            if concept_targets:
                score += len({c.lower() for c in prob.concepts} & concept_targets)
            if sub_targets:
                score += len({s.lower() for s in prob.sub_concepts} & sub_targets)
            if vector_targets:
                cv = prob.concept_vector
                for idx in vector_targets:
                    if 0 <= idx < len(cv) and cv[idx] > 0.0:
                        score += 1.0
            boost[i] = min(1.0, score)

        max_b = float(np.max(boost))
        if max_b > 0:
            boost /= max_b
        return boost

    def _get_observation(self) -> np.ndarray:
        state = self._current_student.get_current_state()
        knowledge_vector = state["knowledge_state"]
        context_features = self._current_student.get_context_features(self.context_dim)
        type_onehot = self._encode_student_type(self._current_student.student_type)
        obs = np.concatenate([knowledge_vector, context_features, type_onehot]).astype(np.float32)
        return obs

    def _encode_student_type(self, student_type: str) -> np.ndarray:
        vec = np.zeros(self.student_type_dim, dtype=np.float32)
        try:
            vec[self._student_types.index(student_type)] = 1.0
        except ValueError:
            pass  # unknown type → all zeros
        return vec

    def _calculate_reward(
        self,
        result: Dict[str, Any],
        student: CognitiveStudentModel,
        problem: Dict[str, Any],
        knowledge_delta: float = 0.0,
    ) -> float:
        """Per-step reward shaping.

        Design rationale (grounded in ITS reward-shaping literature):

        The dominant learning signal is the TERMINAL NLG reward added in step()
        after this call.  Per-step signals here are intentionally kept small so
        they provide behavioral guidance without creating exploitable gradients.

        Key fix  -  give-up arbitrage:
          Old design: give_up = -10 (fixed), wrong step = up to -10.5 (repeating).
          The agent learned that ending episodes immediately minimises cumulative
          cost (confirmed: give-up rate 80% → 100%, ep_length 10 → 1 by step 7000).
          New design: give_up = -40 (must exceed realistic multi-step session cost)
          + terminal NLG penalty (no learning = near-zero terminal bonus).

        Key fix  -  knowledge delta dilution:
          Old: knowledge_delta = mean over ALL 58 concepts → ~0.0008 per step
               → reward += 20 × 0.0008 = +0.016 (negligible vs. -10 give-up)
          New: knowledge_delta passed in = mean over ACTIVE concepts only
               → ~0.01–0.05 per step → reward += 15 × 0.03 = +0.45 (meaningful)

        Key fix  -  accumulated per-step costs:
          Old: needs_more_training = -2.0 per step (fires on ~every failure)
               motivation < 0.40 = -3.0 per step (fires repeatedly at low morale)
          These piled up over 20 steps to make long sessions highly unprofitable.
          New values are reduced so long sessions are never worse than give-up.
        """
        reward = 0.0

        # --- Primary signal: answer correctness ---
        if result.get("answer_was_graded"):
            if result.get("answer_correct") is True:
                reward += 1.5
            else:
                reward -= 1.0
        else:
            if result["success"]:
                reward += 1.0
            else:
                reward -= 0.5

        # --- Knowledge-gain reward (active-concept delta, passed from step()) ---
        if knowledge_delta > 0.0:
            reward += 35.0 * knowledge_delta

        # --- Needs-more-training: soft signal only, reduced from -2.0 ---
        # Was a major contributor to accumulated per-step cost (fires on every
        # failure). Reduced to prevent multi-step sessions from being
        # systematically worse than a single give-up.
        if result.get("needs_more_training"):
            reward -= 0.3
        reward -= 0.2 * float(result.get("mistake_penalty", 0.0))

        # --- Motivation maintenance (softer; terminal drain penalty in step()) ---
        if student.motivation_level > 0.7:
            reward += 0.1
        elif student.motivation_level > 0.5:
            reward += 0.05
        elif student.motivation_level < 0.40:
            reward -= 1.0

        # --- Give-up penalty: must dominate any plausible session cost ---
        # Worst-case 20-step session (all signals negative, motivation < 0.4):
        #   20 × (-0.5 - 0.3 - 0.2 - 1.0 - 1.0) = 20 × (-3.0) = -60
        # In practice students don't stay maximally negative all session;
        # realistic worst case ≈ -25 to -35 over 20 steps.
        # Setting give-up to -40 ensures it is always worse than a failed
        # but genuine session, removing the early-termination incentive.
        if result["gave_up"]:
            reward -= 40.0

        # --- Consecutive-failure streak penalty ---
        # Env-side mechanics (ceiling + difficulty bias) handle the
        # mechanical constraint, but the policy still needs a direct
        # gradient saying "repeated failures on similar problems are
        # costly  -  change strategy."  Without this, the +0.8 ZPD bonus
        # roughly cancels the -1.0 wrong-answer penalty (net -0.2),
        # giving the policy too weak a signal to diversify after stuck
        # patterns.  Empirically remediation_followup_success_rate was
        # 0-7% across all 11 checkpoints.
        consec_fail = int(getattr(student, "consecutive_failures", 0))
        if consec_fail >= 2 and not result["success"]:
            streak_penalty = 0.5 * min(consec_fail - 1, 3)  # -0.5, -1.0, -1.5
            reward -= streak_penalty

        # --- ZPD (Zone of Proximal Development) bonus ---
        # Uses difficulty_score vs K_mean directly  -  consistent with the
        # evaluation framework's zpd_adherence metric (not success_probability proxy).
        # ZPD: problem slightly above student's current level.
        # Vygotsky (1978): optimal challenge zone ≈ [K, K + 0.15].
        _k_mean_reward = float(np.mean(student.knowledge_state))
        _diff_score = float(problem.get("difficulty_score", 0.5))
        _zpd_lo = _k_mean_reward - 0.05   # allow slight below-level consolidation
        _zpd_hi = _k_mean_reward + 0.15   # upper ZPD boundary
        if _zpd_lo <= _diff_score <= _zpd_hi:
            reward += 0.8   # in ZPD
        elif _diff_score < _zpd_lo - 0.10:
            reward -= 0.3   # too easy  -  no learning value
        elif _diff_score > _zpd_hi + 0.10:
            reward -= 1.0   # too hard  -  student cannot engage productively

        # --- Help-seeking is good when genuinely struggling ---
        _confidence = float(result.get("confidence", 0.5))
        if result.get("help_requested", False) and _confidence < 0.4:
            reward += 0.5
        elif result.get("help_requested", False) and not result["success"]:
            reward += 0.2

        return float(reward)

    def render(self):
        return None
