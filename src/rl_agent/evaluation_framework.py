"""
Comprehensive evaluation framework for the personalized tutoring agent.

Research goal
-------------
A shared exam is given to ALL students.  The tutor must improve every
student from their own starting point, adapting pace and content to
each student type.

Grounded in ITS evaluation methodology (Clement et al. 2015; STEP 2020;
Springer ML 2023; RL-DKT 2025).

This module answers six research questions:

  1. Learning effectiveness   -  did every student's knowledge actually grow?
  2. Exam readiness           -  are students ready for the shared exam?
  3. Personalization quality  -  does the tutor behave differently per type?
  4. Engagement               -  is the student motivated and not quitting?
  5. Efficiency               -  is the tutor using problems well?
  6. Baseline comparison      -  does it beat random / naive tutors?

Key metrics (ITS gold standards):
  - NLG  (Normalized Learning Gain): (post − pre) / (1 − pre)
  - Mastery Acquisition Rate:  fraction of concepts reaching ≥ mastery_threshold
  - Time-to-Mastery:           problems needed to cross the threshold
  - ZPD Adherence:             fraction of steps in optimal challenge zone
  - Quartile Analysis:         does the tutor help low performers most?
"""



from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from rl_agent.tutoring_env import SimulatedTutoringEnv


# ---------------------------------------------------------------------------
# Concept-domain index map (mirrors problem_bank.py group layout)
# ---------------------------------------------------------------------------

_CONCEPT_DOMAINS: Dict[str, range] = {
    "Arithmetic":         range(0,  10),
    "Fractions/Decimals": range(10, 18),
    "Percentages/Ratios": range(18, 24),
    "Algebra":            range(24, 30),
    "Geometry":           range(30, 40),
    "Number Theory":      range(40, 44),
    "Statistics/Prob":    range(44, 49),
    "Time/Money":         range(49, 51),
    "Problem-Solving":    range(51, 58),
}


def _compute_nlg(initial: float, final: float) -> float:
    """Normalized Learning Gain = (final − initial) / (1 − initial).

    This is the ITS gold-standard metric (Clement et al. 2015, STEP 2020).
    Clipped to [−1, 1]: negative means the student regressed.
    """
    denom = max(1.0 - initial, 1e-3)
    return float(np.clip((final - initial) / denom, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class PolicyLike(Protocol):
    """Minimal interface for SB3-compatible models and hand-written policies."""

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        ...


ActionFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Full evaluation configuration.

    Parameters
    ----------
    episodes:
        Number of episodes to run for the evaluated policy.
    max_steps:
        Override env.max_steps when set.
    mastery_threshold:
        Mean knowledge level (0–1) considered exam-ready.
    window_splits:
        How many time windows for early-vs-late analysis.
    compare_baselines:
        Whether to also run the four baseline policies.
    baseline_episodes:
        Fewer episodes for baselines to save time.
    capture_concept_vectors:
        Store full concept knowledge vectors per episode (needed for
        per-concept analysis). Adds a small memory overhead.
    """
    episodes: int = 50
    max_steps: Optional[int] = None
    mastery_threshold: float = 0.70
    window_splits: int = 2
    compare_baselines: bool = True
    baseline_episodes: int = 30
    capture_concept_vectors: bool = True


# ---------------------------------------------------------------------------
# Per-step log
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    """One problem attempt within an episode."""
    step: int
    difficulty: float
    success: bool
    gave_up: bool
    knowledge_delta: float
    knowledge_mean: float      # mean(knowledge_state) after this step
    motivation: float
    confidence: float          # student's success probability (ZPD proxy)
    in_zpd: bool               # confidence in [0.35, 0.75]
    needs_more_training: bool


# ---------------------------------------------------------------------------
# Per-episode log
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    """Full per-episode record."""
    student_type: str
    reward: float
    length: int
    successes: int
    gave_up: bool              # any give-up during episode
    initial_knowledge: float   # mean(knowledge_state) at reset
    final_knowledge: float     # mean(knowledge_state) at end
    knowledge_gain: float      # final − initial
    normalized_learning_gain: float   # NLG = (final−initial)/(1−initial)
    mastery_reached: bool      # final_knowledge >= mastery_threshold
    problems_to_mastery: Optional[int]  # None if mastery not reached
    mean_difficulty: float
    mean_knowledge_delta: float
    mean_motivation_delta: float
    mean_answer_correctness: float
    initial_motivation: float
    final_motivation: float
    zpd_adherence_rate: float
    # Full concept vectors  -  populated when capture_concept_vectors=True
    initial_knowledge_vector: List[float] = field(default_factory=list)
    final_knowledge_vector:   List[float] = field(default_factory=list)
    steps: List[StepLog] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Concept-domain analysis
# ---------------------------------------------------------------------------

@dataclass
class ConceptDomainAnalysis:
    """Per-concept and per-domain learning gains.

    Answers: *where* did the tutor most improve the student?
    Which concept domains improved the most / least?
    """
    # Per-concept mean gain (length = concept_dim, e.g. 58)
    per_concept_mean_gain: List[float]
    # Fraction of episodes where each concept reached mastery_threshold
    per_concept_mastery_rate: List[float]
    # Domain-level mean gain (9 domains)
    domain_mean_gain: Dict[str, float]
    # Domain-level mastery rate
    domain_mastery_rate: Dict[str, float]
    # Sorted by gain (best → worst)
    top_improved_domains: List[str]
    least_improved_domains: List[str]


# ---------------------------------------------------------------------------
# Quartile analysis
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeQuartile:
    """Results for one initial-knowledge quartile."""
    quartile: int          # 1=lowest, 4=highest initial knowledge
    n_episodes: int
    initial_knowledge_range: Tuple[float, float]
    mean_initial_knowledge: float
    mean_final_knowledge: float
    mean_knowledge_gain: float
    normalized_learning_gain: float   # NLG
    mastery_rate: float
    give_up_rate: float
    success_rate: float


@dataclass
class QuartileAnalysis:
    """Split students by initial knowledge into 4 quartiles.

    Key finding from Springer ML 2023: RL tutors disproportionately
    benefit lower-performing students.  If Q1.NLG > Q4.NLG then
    `low_performer_advantage` is positive  -  which is the desired equity result.
    """
    quartiles: List[KnowledgeQuartile]
    low_performer_advantage: float   # Q1.NLG − Q4.NLG


# ---------------------------------------------------------------------------
# Aggregate summaries
# ---------------------------------------------------------------------------

@dataclass
class StudentTypeSummary:
    """Per-type aggregate over all episodes of that type."""
    student_type: str
    n_episodes: int
    mean_knowledge_gain: float
    mean_nlg: float
    mean_initial_knowledge: float
    mean_final_knowledge: float
    mastery_rate: float
    success_rate: float
    gave_up_rate: float
    mean_difficulty: float
    mean_motivation_gain: float
    zpd_adherence: float
    mean_problems_to_mastery: float


@dataclass
class PersonalizationSummary:
    """Quantifies HOW WELL the tutor personalizes across student types."""
    zpd_adherence_global: float
    difficulty_knowledge_correlation: float   # Spearman rank corr
    difficulty_differentiation: float         # std across types
    remediation_followup_success_rate: float  # success rate after failure step
    type_summaries: List[StudentTypeSummary]


@dataclass
class ExamReadinessSummary:
    """Simulates how students would perform on the shared exam."""
    mastery_threshold: float
    fraction_ready: float
    mean_final_knowledge: float
    std_final_knowledge: float
    mean_nlg: float
    knowledge_gain_vs_initial: float
    fraction_ready_by_type: Dict[str, float]
    mean_problems_to_mastery: float


@dataclass
class BaselineResult:
    """Summary for one baseline policy."""
    name: str
    description: str
    mean_knowledge_gain: float
    mean_nlg: float
    fraction_exam_ready: float
    mean_reward: float
    gave_up_rate: float
    zpd_adherence: float
    mean_difficulty: float


@dataclass
class EvaluationSummary:
    """Top-level evaluation output covering all six research questions.

    Call ``summary.print_report()`` for a human-readable research report,
    or ``summary.to_dict()`` for JSON serialisation.
    """
    config: EvaluationConfig

    # Q1  -  Learning effectiveness
    mean_knowledge_gain: float
    std_knowledge_gain: float
    mean_nlg: float
    all_types_improved: bool
    nlg_by_type: Dict[str, float]

    # Q2  -  Exam readiness
    exam_readiness: ExamReadinessSummary

    # Q3  -  Personalization
    personalization: PersonalizationSummary

    # Q4  -  Engagement
    mean_motivation_start: float
    mean_motivation_end: float
    give_up_rate: float
    mean_reward: float
    std_reward: float

    # Q5  -  Efficiency
    mean_problems_to_mastery: float
    success_rate: float

    # Q6  -  Baselines
    baselines: List[BaselineResult] = field(default_factory=list)

    # Extras
    windows: List[Dict[str, float]] = field(default_factory=list)
    quartile_analysis: Optional[QuartileAnalysis] = None
    concept_analysis: Optional[ConceptDomainAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["baselines"] = [asdict(b) for b in self.baselines]
        return data

    def print_report(self, title: str = "PERSONALIZED TUTOR EVALUATION") -> None:
        """Print a human-readable, research-grade report to stdout."""
        sep = "=" * 66
        print(sep)
        print(f"  {title}")
        print(sep)

        # --- Q1: Learning Effectiveness ---
        print("\n[Q1  -  Learning Effectiveness (NLG = Normalized Learning Gain)]")
        print(f"  Mean knowledge gain : {self.mean_knowledge_gain:+.4f} ± {self.std_knowledge_gain:.4f}")
        print(f"  Mean NLG            : {self.mean_nlg:+.4f}  "
              "(target > 0.20 for good tutor)")
        print(f"  All types improved  : {'YES ✓' if self.all_types_improved else 'NO ✗'}")
        print(f"\n  {'Student type':<22} {'Gain':>7}  {'NLG':>7}  {'Init':>6}  {'Final':>6}")
        print("  " + "-" * 55)
        for ts in self.personalization.type_summaries:
            nlg = self.nlg_by_type.get(ts.student_type, float("nan"))
            print(f"  {ts.student_type:<22} {ts.mean_knowledge_gain:>+7.4f}  "
                  f"{nlg:>+7.4f}  {ts.mean_initial_knowledge:>6.3f}  "
                  f"{ts.mean_final_knowledge:>6.3f}")

        # --- Q2: Exam Readiness ---
        print(f"\n[Q2  -  Exam Readiness]  (mastery ≥ {self.exam_readiness.mastery_threshold:.0%})")
        print(f"  Exam-ready students : {self.exam_readiness.fraction_ready:.1%}")
        print(f"  Mean NLG            : {self.exam_readiness.mean_nlg:+.4f}")
        print(f"  Knowledge gain      : {self.exam_readiness.knowledge_gain_vs_initial:+.4f}")
        print(f"  Mean final knowledge: {self.exam_readiness.mean_final_knowledge:.3f} "
              f"± {self.exam_readiness.std_final_knowledge:.3f}")
        print(f"  Probs to mastery    : {self.exam_readiness.mean_problems_to_mastery:.1f}")
        print(f"\n  {'Student type':<22} {'Exam-ready':>10}")
        print("  " + "-" * 34)
        for stype, rate in self.exam_readiness.fraction_ready_by_type.items():
            print(f"  {stype:<22} {rate:>10.1%}")

        # --- Q3: Personalization ---
        print("\n[Q3  -  Personalization Quality]")
        p = self.personalization
        print(f"  ZPD adherence       : {p.zpd_adherence_global:.1%}  "
              "(target > 50%)")
        print(f"  Difficulty ~ knowledge corr: {p.difficulty_knowledge_correlation:+.3f}  "
              "(target > 0.30  -  tutor tracks growth)")
        print(f"  Diff. differentiation (std): {p.difficulty_differentiation:.4f}  "
              "(higher → more per-type adaptation)")
        print(f"  Remediation success rate   : {p.remediation_followup_success_rate:.1%}")
        print(f"\n  {'Student type':<22} {'Diff':>6}  {'ZPD':>6}  {'GiveUp':>7}  {'Mastery':>8}")
        print("  " + "-" * 55)
        for ts in p.type_summaries:
            print(f"  {ts.student_type:<22} {ts.mean_difficulty:>6.3f}  "
                  f"{ts.zpd_adherence:>6.1%}  {ts.gave_up_rate:>7.1%}  "
                  f"{ts.mastery_rate:>8.1%}")

        # --- Q4: Engagement ---
        print("\n[Q4  -  Engagement]")
        print(f"  Give-up rate        : {self.give_up_rate:.1%}")
        print(f"  Motivation start    : {self.mean_motivation_start:.3f}")
        print(f"  Motivation end      : {self.mean_motivation_end:.3f}  "
              f"(delta={self.mean_motivation_end - self.mean_motivation_start:+.4f})")
        print(f"  Mean reward         : {self.mean_reward:.2f} ± {self.std_reward:.2f}")

        # --- Q5: Efficiency ---
        print("\n[Q5  -  Efficiency]")
        print(f"  Success rate        : {self.success_rate:.1%}")
        print(f"  Mean probs to master: {self.mean_problems_to_mastery:.1f}")

        # --- Quartile analysis ---
        if self.quartile_analysis:
            print("\n[Quartile Analysis  -  by Initial Knowledge]")
            print(f"  Low-performer advantage: {self.quartile_analysis.low_performer_advantage:+.4f}  "
                  "(positive = RL helps weak students more)")
            print(f"  {'Quartile':<12} {'Init range':>16}  {'Gain':>7}  {'NLG':>7}  "
                  f"{'Mastery':>8}  {'GiveUp':>7}")
            print("  " + "-" * 64)
            for q in self.quartile_analysis.quartiles:
                r = q.initial_knowledge_range
                print(f"  Q{q.quartile} (n={q.n_episodes:<4}) "
                      f"[{r[0]:.2f}, {r[1]:.2f}]  "
                      f"{q.mean_knowledge_gain:>+7.4f}  "
                      f"{q.normalized_learning_gain:>+7.4f}  "
                      f"{q.mastery_rate:>8.1%}  "
                      f"{q.give_up_rate:>7.1%}")

        # --- Concept domain analysis ---
        if self.concept_analysis:
            print("\n[Concept Domain Analysis  -  Mean Knowledge Gain]")
            sorted_domains = sorted(
                self.concept_analysis.domain_mean_gain.items(),
                key=lambda x: x[1], reverse=True
            )
            print(f"  {'Domain':<22} {'Gain':>7}  {'Mastery Rate':>13}")
            print("  " + "-" * 46)
            for domain, gain in sorted_domains:
                mr = self.concept_analysis.domain_mastery_rate.get(domain, 0.0)
                bar = "█" * max(0, int(gain * 200)) + "░" * max(0, 10 - int(gain * 200))
                print(f"  {domain:<22} {gain:>+7.4f}  {mr:>13.1%}  {bar}")

        # --- Time windows ---
        if self.windows:
            print("\n[Time Windows  -  Early vs Late Episodes]")
            for i, w in enumerate(self.windows):
                label = f"Window {i+1} (ep {int(w['start_episode'])}–{int(w['end_episode'])})"
                print(f"  {label}")
                print(f"    NLG={w.get('mean_nlg', 0):+.4f}  "
                      f"reward={w['mean_reward']:.2f}  "
                      f"gain={w['mean_knowledge_delta']:+.4f}  "
                      f"give_up={w['gave_up_rate']:.1%}")

        # --- Baseline comparison ---
        if self.baselines:
            print("\n[Q6  -  Baseline Comparison]")
            hdr = (f"  {'Policy':<24} {'Gain':>7}  {'NLG':>7}  "
                   f"{'Ready':>7}  {'Reward':>8}  {'GiveUp':>8}  {'ZPD':>6}")
            print(hdr)
            print("  " + "-" * 66)
            print(f"  {'>>> this policy <<<':<24} "
                  f"{self.mean_knowledge_gain:>+7.4f}  "
                  f"{self.mean_nlg:>+7.4f}  "
                  f"{self.exam_readiness.fraction_ready:>7.1%}  "
                  f"{self.mean_reward:>8.2f}  "
                  f"{self.give_up_rate:>8.1%}  "
                  f"{self.personalization.zpd_adherence_global:>6.1%}")
            for b in self.baselines:
                print(f"  {b.name:<24} "
                      f"{b.mean_knowledge_gain:>+7.4f}  "
                      f"{b.mean_nlg:>+7.4f}  "
                      f"{b.fraction_exam_ready:>7.1%}  "
                      f"{b.mean_reward:>8.2f}  "
                      f"{b.gave_up_rate:>8.1%}  "
                      f"{b.zpd_adherence:>6.1%}")

        print(sep)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def _run_single_policy(
    env: SimulatedTutoringEnv,
    policy_fn: ActionFn,
    config: EvaluationConfig,
    probe_students: Optional[List[Any]] = None,
) -> List[EpisodeLog]:
    """Run `policy_fn` for `config.episodes` episodes and collect detailed logs."""
    logs: List[EpisodeLog] = []
    max_steps = config.max_steps or env.max_steps
    concept_dim = env.concept_dim

    episode_students = probe_students if probe_students is not None else [None] * config.episodes

    for fixed_student in episode_students:
        reset_options = {"probe_student": fixed_student} if fixed_student is not None else None
        result = env.reset(options=reset_options)
        obs = result[0] if isinstance(result, tuple) else result

        # Snapshot initial state from observation
        # obs layout: [knowledge(concept_dim), context(16), type_onehot(7)]
        initial_knowledge_vector = obs[:concept_dim].tolist() if config.capture_concept_vectors else []
        initial_knowledge = float(np.mean(obs[:concept_dim]))
        initial_motivation = float(obs[concept_dim])   # context[0] = motivation
        student_type = env._current_student.student_type

        done = False
        ep_reward = 0.0
        ep_successes = 0
        ep_gave_up = False
        mastery_reached = False
        problems_to_mastery: Optional[int] = None
        step_logs: List[StepLog] = []
        difficulties: List[float] = []
        knowledge_deltas: List[float] = []
        motivation_deltas: List[float] = []
        correctness_flags: List[float] = []
        remediation_pending = False

        step = 0
        while not done:
            action = policy_fn(np.asarray(obs, dtype=np.float32))
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_result

            step += 1
            ep_reward += float(reward)

            success  = bool(info.get("success", False))
            gave_up  = bool(info.get("gave_up", False))
            difficulty = float(info.get("difficulty", 0.0))
            k_delta    = float(info.get("knowledge_delta", 0.0))
            m_delta    = float(info.get("motivation_delta", 0.0))
            confidence = float(info.get("confidence", 0.5))
            motivation = float(info.get("motivation", initial_motivation))
            needs_more = bool(info.get("needs_more_training", False))
            answer_correct   = info.get("answer_correct")
            answer_was_graded = info.get("answer_was_graded", False)

            if success:
                ep_successes += 1
            if gave_up:
                ep_gave_up = True

            in_zpd = 0.35 <= confidence <= 0.75
            current_knowledge = float(np.mean(obs[:concept_dim]))

            step_logs.append(StepLog(
                step=step,
                difficulty=difficulty,
                success=success,
                gave_up=gave_up,
                knowledge_delta=k_delta,
                knowledge_mean=current_knowledge,
                motivation=motivation,
                confidence=confidence,
                in_zpd=in_zpd,
                needs_more_training=needs_more,
            ))

            difficulties.append(difficulty)
            knowledge_deltas.append(k_delta)
            motivation_deltas.append(m_delta)

            if answer_was_graded:
                correctness_flags.append(1.0 if answer_correct else 0.0)

            # Remediation tracking
            if remediation_pending:
                remediation_pending = False   # consumed; don't track beyond 1 step
            if needs_more and not success:
                remediation_pending = True

            if not mastery_reached and current_knowledge >= config.mastery_threshold:
                mastery_reached = True
                problems_to_mastery = step

            if step >= max_steps:
                done = True

        final_knowledge = float(np.mean(obs[:concept_dim]))
        final_motivation = float(obs[concept_dim])
        final_knowledge_vector = obs[:concept_dim].tolist() if config.capture_concept_vectors else []

        nlg = _compute_nlg(initial_knowledge, final_knowledge)

        logs.append(EpisodeLog(
            student_type=student_type,
            reward=ep_reward,
            length=step,
            successes=ep_successes,
            gave_up=ep_gave_up,
            initial_knowledge=initial_knowledge,
            final_knowledge=final_knowledge,
            knowledge_gain=final_knowledge - initial_knowledge,
            normalized_learning_gain=nlg,
            mastery_reached=mastery_reached,
            problems_to_mastery=problems_to_mastery,
            mean_difficulty=float(np.mean(difficulties)) if difficulties else 0.0,
            mean_knowledge_delta=float(np.mean(knowledge_deltas)) if knowledge_deltas else 0.0,
            mean_motivation_delta=float(np.mean(motivation_deltas)) if motivation_deltas else 0.0,
            mean_answer_correctness=float(np.mean(correctness_flags)) if correctness_flags else 0.0,
            initial_motivation=initial_motivation,
            final_motivation=final_motivation,
            zpd_adherence_rate=float(np.mean([s.in_zpd for s in step_logs])) if step_logs else 0.0,
            initial_knowledge_vector=initial_knowledge_vector,
            final_knowledge_vector=final_knowledge_vector,
            steps=step_logs,
        ))

    return logs


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _per_type_summary(
    logs: List[EpisodeLog],
    student_type: str,
    mastery_threshold: float,
) -> StudentTypeSummary:
    tl = [lg for lg in logs if lg.student_type == student_type]
    if not tl:
        return StudentTypeSummary(
            student_type=student_type, n_episodes=0,
            mean_knowledge_gain=0.0, mean_nlg=0.0,
            mean_initial_knowledge=0.0, mean_final_knowledge=0.0,
            mastery_rate=0.0, success_rate=0.0, gave_up_rate=0.0,
            mean_difficulty=0.0, mean_motivation_gain=0.0,
            zpd_adherence=0.0, mean_problems_to_mastery=float("nan"),
        )
    total_steps = sum(lg.length for lg in tl) or 1
    ptm = [lg.problems_to_mastery for lg in tl if lg.problems_to_mastery is not None]
    return StudentTypeSummary(
        student_type=student_type,
        n_episodes=len(tl),
        mean_knowledge_gain=float(np.mean([lg.knowledge_gain for lg in tl])),
        mean_nlg=float(np.mean([lg.normalized_learning_gain for lg in tl])),
        mean_initial_knowledge=float(np.mean([lg.initial_knowledge for lg in tl])),
        mean_final_knowledge=float(np.mean([lg.final_knowledge for lg in tl])),
        mastery_rate=float(np.mean([1.0 if lg.mastery_reached else 0.0 for lg in tl])),
        success_rate=float(sum(lg.successes for lg in tl) / total_steps),
        gave_up_rate=float(np.mean([1.0 if lg.gave_up else 0.0 for lg in tl])),
        mean_difficulty=float(np.mean([lg.mean_difficulty for lg in tl])),
        mean_motivation_gain=float(np.mean([lg.final_motivation - lg.initial_motivation for lg in tl])),
        zpd_adherence=float(np.mean([lg.zpd_adherence_rate for lg in tl])),
        mean_problems_to_mastery=float(np.mean(ptm)) if ptm else float("nan"),
    )


def _compute_personalization(
    logs: List[EpisodeLog],
    mastery_threshold: float,
) -> PersonalizationSummary:
    all_types = sorted({lg.student_type for lg in logs})
    type_summaries = [_per_type_summary(logs, t, mastery_threshold) for t in all_types]

    all_steps = [s for lg in logs for s in lg.steps]
    zpd_global = float(np.mean([s.in_zpd for s in all_steps])) if all_steps else 0.0

    knowledges  = np.array([lg.initial_knowledge for lg in logs])
    difficulties = np.array([lg.mean_difficulty for lg in logs])
    diff_knowledge_corr = 0.0
    if len(knowledges) > 2 and np.std(knowledges) > 0 and np.std(difficulties) > 0:
        krank = np.argsort(np.argsort(knowledges)).astype(float)
        drank = np.argsort(np.argsort(difficulties)).astype(float)
        diff_knowledge_corr = float(np.corrcoef(krank, drank)[0, 1])

    type_mean_diffs = [ts.mean_difficulty for ts in type_summaries if ts.n_episodes > 0]
    diff_differentiation = float(np.std(type_mean_diffs)) if len(type_mean_diffs) > 1 else 0.0

    remediation_successes: List[int] = []
    for lg in logs:
        for i, s in enumerate(lg.steps[:-1]):
            if s.needs_more_training and not s.success:
                next_s = lg.steps[i + 1]
                remediation_successes.append(1 if next_s.success and not next_s.gave_up else 0)
    remediation_rate = float(np.mean(remediation_successes)) if remediation_successes else float("nan")

    return PersonalizationSummary(
        zpd_adherence_global=zpd_global,
        difficulty_knowledge_correlation=diff_knowledge_corr,
        difficulty_differentiation=diff_differentiation,
        remediation_followup_success_rate=remediation_rate,
        type_summaries=type_summaries,
    )


def _compute_exam_readiness(
    logs: List[EpisodeLog],
    mastery_threshold: float,
) -> ExamReadinessSummary:
    all_types = sorted({lg.student_type for lg in logs})
    ptm = [lg.problems_to_mastery for lg in logs if lg.problems_to_mastery is not None]
    return ExamReadinessSummary(
        mastery_threshold=mastery_threshold,
        fraction_ready=float(np.mean([1.0 if lg.mastery_reached else 0.0 for lg in logs])),
        mean_final_knowledge=float(np.mean([lg.final_knowledge for lg in logs])),
        std_final_knowledge=float(np.std([lg.final_knowledge for lg in logs])),
        mean_nlg=float(np.mean([lg.normalized_learning_gain for lg in logs])),
        knowledge_gain_vs_initial=(
            float(np.mean([lg.final_knowledge for lg in logs]))
            - float(np.mean([lg.initial_knowledge for lg in logs]))
        ),
        fraction_ready_by_type={
            t: float(np.mean([1.0 if lg.mastery_reached else 0.0
                               for lg in logs if lg.student_type == t]) or 0.0)
            for t in all_types
        },
        mean_problems_to_mastery=float(np.mean(ptm)) if ptm else float("nan"),
    )


def _compute_quartile_analysis(logs: List[EpisodeLog]) -> QuartileAnalysis:
    """Split episodes by initial knowledge into 4 quartiles."""
    initial_vals = [lg.initial_knowledge for lg in logs]
    quartile_boundaries = np.percentile(initial_vals, [0, 25, 50, 75, 100])

    quartiles: List[KnowledgeQuartile] = []
    for q in range(4):
        lo = float(quartile_boundaries[q])
        hi = float(quartile_boundaries[q + 1])
        # Include hi endpoint in last quartile; use < for others
        if q < 3:
            ql = [lg for lg in logs if lo <= lg.initial_knowledge < hi]
        else:
            ql = [lg for lg in logs if lo <= lg.initial_knowledge <= hi]

        if not ql:
            quartiles.append(KnowledgeQuartile(
                quartile=q + 1, n_episodes=0,
                initial_knowledge_range=(lo, hi),
                mean_initial_knowledge=lo,
                mean_final_knowledge=lo,
                mean_knowledge_gain=0.0,
                normalized_learning_gain=0.0,
                mastery_rate=0.0,
                give_up_rate=0.0,
                success_rate=0.0,
            ))
            continue

        total_steps = sum(lg.length for lg in ql) or 1
        quartiles.append(KnowledgeQuartile(
            quartile=q + 1,
            n_episodes=len(ql),
            initial_knowledge_range=(lo, hi),
            mean_initial_knowledge=float(np.mean([lg.initial_knowledge for lg in ql])),
            mean_final_knowledge=float(np.mean([lg.final_knowledge for lg in ql])),
            mean_knowledge_gain=float(np.mean([lg.knowledge_gain for lg in ql])),
            normalized_learning_gain=float(np.mean([lg.normalized_learning_gain for lg in ql])),
            mastery_rate=float(np.mean([1.0 if lg.mastery_reached else 0.0 for lg in ql])),
            give_up_rate=float(np.mean([1.0 if lg.gave_up else 0.0 for lg in ql])),
            success_rate=float(sum(lg.successes for lg in ql) / total_steps),
        ))

    q1_nlg = quartiles[0].normalized_learning_gain if quartiles else 0.0
    q4_nlg = quartiles[3].normalized_learning_gain if len(quartiles) >= 4 else 0.0
    return QuartileAnalysis(
        quartiles=quartiles,
        low_performer_advantage=q1_nlg - q4_nlg,
    )


def _compute_concept_analysis(
    logs: List[EpisodeLog],
    mastery_threshold: float,
) -> Optional[ConceptDomainAnalysis]:
    """Per-concept gain analysis. Requires capture_concept_vectors=True."""
    logs_with_vectors = [
        lg for lg in logs
        if lg.initial_knowledge_vector and lg.final_knowledge_vector
        and len(lg.initial_knowledge_vector) == len(lg.final_knowledge_vector)
    ]
    if not logs_with_vectors:
        return None

    concept_dim = len(logs_with_vectors[0].initial_knowledge_vector)
    init_matrix = np.array([lg.initial_knowledge_vector for lg in logs_with_vectors])
    final_matrix = np.array([lg.final_knowledge_vector for lg in logs_with_vectors])
    gain_matrix  = final_matrix - init_matrix

    per_concept_mean_gain  = gain_matrix.mean(axis=0).tolist()
    per_concept_mastery_rate = (final_matrix >= mastery_threshold).mean(axis=0).tolist()

    # Domain aggregates  -  use only dims within concept_dim
    domain_mean_gain: Dict[str, float] = {}
    domain_mastery_rate: Dict[str, float] = {}
    for domain, idx_range in _CONCEPT_DOMAINS.items():
        valid_dims = [i for i in idx_range if i < concept_dim]
        if not valid_dims:
            continue
        domain_mean_gain[domain] = float(gain_matrix[:, valid_dims].mean())
        domain_mastery_rate[domain] = float((final_matrix[:, valid_dims] >= mastery_threshold).mean())

    sorted_domains = sorted(domain_mean_gain.keys(), key=lambda d: domain_mean_gain[d], reverse=True)

    return ConceptDomainAnalysis(
        per_concept_mean_gain=per_concept_mean_gain,
        per_concept_mastery_rate=per_concept_mastery_rate,
        domain_mean_gain=domain_mean_gain,
        domain_mastery_rate=domain_mastery_rate,
        top_improved_domains=sorted_domains[:3],
        least_improved_domains=sorted_domains[-3:][::-1],
    )


def _window_analysis(logs: List[EpisodeLog], window_splits: int) -> List[Dict[str, float]]:
    windows: List[Dict[str, float]] = []
    splits  = max(1, window_splits)
    indices = np.linspace(0, len(logs), splits + 1, dtype=int)
    for i in range(splits):
        start, end = int(indices[i]), int(indices[i + 1])
        if end <= start:
            continue
        w = logs[start:end]
        total_steps = sum(lg.length for lg in w) or 1
        windows.append({
            "start_episode":      float(start),
            "end_episode":        float(end),
            "mean_reward":        float(np.mean([lg.reward for lg in w])),
            "mean_length":        float(np.mean([lg.length for lg in w])),
            "success_rate":       float(sum(lg.successes for lg in w) / total_steps),
            "gave_up_rate":       float(np.mean([1.0 if lg.gave_up else 0.0 for lg in w])),
            "mean_difficulty":    float(np.mean([lg.mean_difficulty for lg in w])),
            "mean_knowledge_delta": float(np.mean([lg.mean_knowledge_delta for lg in w])),
            "mean_knowledge_gain":  float(np.mean([lg.knowledge_gain for lg in w])),
            "mean_nlg":             float(np.mean([lg.normalized_learning_gain for lg in w])),
            "mean_motivation_delta": float(np.mean([lg.mean_motivation_delta for lg in w])),
            "zpd_adherence":        float(np.mean([lg.zpd_adherence_rate for lg in w])),
        })
    return windows


def _summarise_logs(
    logs: List[EpisodeLog],
    config: EvaluationConfig,
) -> EvaluationSummary:
    total_steps = sum(lg.length for lg in logs) or 1
    ptm = [lg.problems_to_mastery for lg in logs if lg.problems_to_mastery is not None]
    gains = [lg.knowledge_gain for lg in logs]
    nlgs  = [lg.normalized_learning_gain for lg in logs]

    all_types = sorted({lg.student_type for lg in logs})
    nlg_by_type = {
        t: float(np.mean([lg.normalized_learning_gain for lg in logs if lg.student_type == t]))
        for t in all_types
    }

    personalization = _compute_personalization(logs, config.mastery_threshold)
    quartile_analysis = _compute_quartile_analysis(logs) if len(logs) >= 4 else None
    concept_analysis  = _compute_concept_analysis(logs, config.mastery_threshold) if config.capture_concept_vectors else None

    return EvaluationSummary(
        config=config,
        mean_knowledge_gain=float(np.mean(gains)),
        std_knowledge_gain=float(np.std(gains)),
        mean_nlg=float(np.mean(nlgs)),
        all_types_improved=all(
            float(np.mean([lg.knowledge_gain for lg in logs if lg.student_type == t])) > 0
            for t in all_types
        ),
        nlg_by_type=nlg_by_type,
        exam_readiness=_compute_exam_readiness(logs, config.mastery_threshold),
        personalization=personalization,
        mean_motivation_start=float(np.mean([lg.initial_motivation for lg in logs])),
        mean_motivation_end=float(np.mean([lg.final_motivation for lg in logs])),
        give_up_rate=float(np.mean([1.0 if lg.gave_up else 0.0 for lg in logs])),
        mean_reward=float(np.mean([lg.reward for lg in logs])),
        std_reward=float(np.std([lg.reward for lg in logs])),
        success_rate=float(sum(lg.successes for lg in logs) / total_steps),
        mean_problems_to_mastery=float(np.mean(ptm)) if ptm else float("nan"),
        windows=_window_analysis(logs, config.window_splits),
        quartile_analysis=quartile_analysis,
        concept_analysis=concept_analysis,
        baselines=[],
    )


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

def _random_policy(env: SimulatedTutoringEnv) -> ActionFn:
    """Uniformly random action (no personalization  -  absolute floor)."""
    def _fn(_obs: np.ndarray) -> np.ndarray:
        return env.action_space.sample()
    return _fn


def _fixed_medium_policy(env: SimulatedTutoringEnv) -> ActionFn:
    """Fixed difficulty=0.55, uniform concepts (ignores student completely)."""
    action = np.zeros(env.action_dim, dtype=np.float32)
    action[:env.concept_dim] = 1.0 / max(env.concept_dim, 1)
    action[env.concept_dim] = 0.55   # meta[0] = difficulty
    norm = np.linalg.norm(action)
    if norm > 0:
        action /= norm
    action_fixed = action.copy()

    def _fn(_obs: np.ndarray) -> np.ndarray:
        return action_fixed.copy()
    return _fn


def _zpd_oracle_policy(env: SimulatedTutoringEnv) -> ActionFn:
    """ZPD oracle: targets weakest concepts at student's current mastery level.

    This is the strongest non-RL baseline.  It reads the student's knowledge
    state directly from the observation and picks the most personalized action
    possible without learned behaviour.  The RL agent should aim to match or
    beat this.
    """
    def _fn(obs: np.ndarray) -> np.ndarray:
        knowledge = np.clip(obs[:env.concept_dim], 0.0, 1.0)
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[:env.concept_dim] = np.clip(1.0 - knowledge, 0.0, 1.0)
        action[env.concept_dim] = float(np.mean(knowledge))
        norm = np.linalg.norm(action)
        if norm > 0:
            action /= norm
        return action
    return _fn


class _AscendingDifficultyPolicy:
    """Linearly increases difficulty 0.30→0.85 over the episode.

    Represents a textbook-style curriculum with no individual adaptation.
    """
    def __init__(self, env: SimulatedTutoringEnv) -> None:
        self.env = env

    def __call__(self, _obs: np.ndarray) -> np.ndarray:
        max_steps = self.env.max_steps or 20
        t = min(self.env._step_count / max(max_steps - 1, 1), 1.0)
        difficulty = 0.30 + 0.55 * t
        action = np.zeros(self.env.action_dim, dtype=np.float32)
        action[:self.env.concept_dim] = 1.0 / max(self.env.concept_dim, 1)
        action[self.env.concept_dim] = difficulty
        norm = np.linalg.norm(action)
        if norm > 0:
            action /= norm
        return action


class _LowestMasteryFirstPolicy:
    """Rule-based: always targets the concept with the lowest current mastery.

    Equivalent to a 'greedy remediation' teacher who fills the biggest gap
    every session.  Stronger than ZPD oracle for some student types.
    """
    def __init__(self, env: SimulatedTutoringEnv) -> None:
        self.env = env

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        knowledge = np.clip(obs[:self.env.concept_dim], 0.0, 1.0)
        action = np.zeros(self.env.action_dim, dtype=np.float32)
        # Concentrate weight on the single weakest concept
        weakest_idx = int(np.argmin(knowledge))
        action[weakest_idx] = 1.0
        # Set difficulty to the weakest concept's current mastery
        action[self.env.concept_dim] = float(knowledge[weakest_idx])
        norm = np.linalg.norm(action)
        if norm > 0:
            action /= norm
        return action


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_policy(
    env: SimulatedTutoringEnv,
    policy: PolicyLike,
    config: Optional[EvaluationConfig] = None,
    probe_students: Optional[List[Any]] = None,
) -> EvaluationSummary:
    """Evaluate a tutoring policy against all six research criteria.

    Parameters
    ----------
    env:
        A configured ``SimulatedTutoringEnv`` instance (the TEST environment  -
        use a different seed from the training environment).
    policy:
        Any object with a ``predict(obs, deterministic)`` method (SB3 PPO, etc.).
    config:
        Evaluation config.  ``EvaluationConfig()`` defaults used when omitted.
    probe_students:
        Optional fixed student list to evaluate instead of sampling random
        students from the environment population. When provided, the same list
        is reused for the main policy and all baselines so per-type comparisons
        are directly comparable.

    Returns
    -------
    EvaluationSummary
        Full evaluation including NLG, per-concept analysis, quartile analysis,
        and baseline comparison.  Call ``summary.print_report()`` for output.
    """
    cfg = config or EvaluationConfig()

    def _policy_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(obs, deterministic=True)  # type: ignore[call-arg]
        return np.asarray(action, dtype=np.float32).reshape(-1)

    logs = _run_single_policy(env, _policy_fn, cfg, probe_students=probe_students)
    summary = _summarise_logs(logs, cfg)

    if cfg.compare_baselines:
        baseline_cfg = EvaluationConfig(
            episodes=len(probe_students) if probe_students is not None else cfg.baseline_episodes,
            max_steps=cfg.max_steps,
            mastery_threshold=cfg.mastery_threshold,
            compare_baselines=False,
            capture_concept_vectors=False,   # faster for baselines
        )
        baselines_spec = [
            ("random",                "Uniformly random (no personalization)",         _random_policy(env)),
            ("fixed_medium",          "Fixed difficulty=0.55, uniform concepts",        _fixed_medium_policy(env)),
            ("zpd_oracle",            "ZPD oracle  -  targets weakest concepts",          _zpd_oracle_policy(env)),
            ("ascending_difficulty",  "Linear ramp 0.30→0.85 (textbook curriculum)",    _AscendingDifficultyPolicy(env)),
            ("lowest_mastery_first",  "Rule-based: always fill the biggest concept gap", _LowestMasteryFirstPolicy(env)),
        ]
        for name, description, policy_fn in baselines_spec:
            b_logs = _run_single_policy(
                env,
                policy_fn,
                baseline_cfg,
                probe_students=probe_students,
            )
            b_summ = _summarise_logs(b_logs, baseline_cfg)
            all_steps = [s for lg in b_logs for s in lg.steps]
            zpd = float(np.mean([s.in_zpd for s in all_steps])) if all_steps else 0.0
            summary.baselines.append(BaselineResult(
                name=name,
                description=description,
                mean_knowledge_gain=b_summ.mean_knowledge_gain,
                mean_nlg=b_summ.mean_nlg,
                fraction_exam_ready=b_summ.exam_readiness.fraction_ready,
                mean_reward=b_summ.mean_reward,
                gave_up_rate=b_summ.give_up_rate,
                zpd_adherence=zpd,
                mean_difficulty=float(np.mean([lg.mean_difficulty for lg in b_logs])),
            ))

    return summary


__all__ = [
    "EvaluationConfig",
    "EpisodeLog",
    "StepLog",
    "StudentTypeSummary",
    "PersonalizationSummary",
    "ExamReadinessSummary",
    "ConceptDomainAnalysis",
    "KnowledgeQuartile",
    "QuartileAnalysis",
    "BaselineResult",
    "EvaluationSummary",
    "evaluate_policy",
]
