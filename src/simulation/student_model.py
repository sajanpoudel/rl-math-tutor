"""
Cognitive student models for simulating realistic learning behaviors.

Research-grounded parameter calibration
=======================================
All core cognitive parameters are calibrated from published empirical work:

Knowledge state (K₀):
    Koedinger, K. R., Carvalho, P. F., Liu, R., & McLaughlin, E. A. (2023).
    "An Astonishing Regularity in Student Learning Rate." PNAS.
    - Mean initial performance ≈ 65% accuracy (range 55–75% across students)
    - KC-wise values drawn around each student's mean

Learning rate (μ):
    Same Koedinger et al. (2023):
    - Typical learning rate ≈ +0.1 log-odds per practice opportunity
    - Translates to ~+2.5% absolute accuracy per opportunity
    - Learning rates are "astonishingly similar" across students (low variance
      relative to initial knowledge)
    - Typical student needs ~7 opportunities to master a knowledge component
    - Update rule uses logit space: logit(K_new) = logit(K_old) + μ

Attention span / mind-wandering:
    Smallwood, J. & Schooler, J. W. (2013). "The restless mind."
    Frontiers in Psychology, 4, 495.
    - Initial attention breaks after ~10–18 min of instruction
    - Late-session lapse frequency: every ~3–4 min
    - Baseline mind-wandering probability: ~30% (modulated by motivation)
    - Mapped to problem counts assuming ~3 min/problem, with 1.5× multiplier
      for interactive tutoring vs. passive lecture

Fatigue (ψ):
    Bayne, H. B. & Inan, F. A.  -  Student Mental Fatigue Survey (SMFS):
    - SMFS Mental Fatigue mean = 2.79/5, SD = 0.92
    - Session-start baseline: small non-zero (0.02–0.12)
    Springer Link (2023)  -  online teaching fatigue incidence: ~65%

Motivation (φ):
    Academic Motivation Scale (AMS)  -  Vallerand et al.:
    - 7 subscales (1–7 scale): IM-Knowledge M=5.17 SD=1.37,
      IM-Accomplishment M=4.65 SD=1.45, IM-Stimulation M=4.93 SD=1.60,
      EM-Identified M=6.05 SD=1.11, EM-Introjected M=5.31 SD=1.37,
      EM-External M=5.85 SD=1.36, Amotivation M=2.08 SD=1.46
    - Collapsed to single φ ∈ [0,1] via weighted composite (population mean ≈ 0.68)
"""

import numpy as np
import random
from collections import Counter, defaultdict, deque
from typing import Any, Deque, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StudentKnowledgeGraph:
    """
    Lightweight per-student knowledge graph.

    Graph schema (adapted to math tutoring):
    - Nodes: question, concept, sub_concept
    - Edge relation types: Prerequisite_of, Used_for, Hyponym_of, Part_of
    - Each concept/sub-concept stores a local cognitive state library with "Good"/"Bad" labels
    """

    RELATION_TYPES = ("Prerequisite_of", "Used_for", "Hyponym_of", "Part_of")

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edge_votes: Dict[tuple, Counter] = defaultdict(Counter)
        self.edges: Dict[tuple, str] = {}
        self.local_state_library: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.question_to_knowledge_nodes: Dict[str, set] = defaultdict(set)

    def ensure_node(self, node_id: str, node_type: str, label: str) -> None:
        self.nodes.setdefault(node_id, {"type": node_type, "label": label})

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        if relation not in self.RELATION_TYPES:
            return
        key = (src, dst)
        self.edge_votes[key][relation] += 1
        self.edges[key] = self.edge_votes[key].most_common(1)[0][0]

    def register_problem(
        self,
        problem: Dict[str, Any],
        concept_names: List[str],
        sub_concept_names: List[str],
    ) -> str:
        question_id = f"question::{problem.get('id', 'unknown')}"
        question_label = str(problem.get("question", f"question_{problem.get('id', 'unknown')}"))
        self.ensure_node(question_id, "question", question_label)

        for concept in concept_names:
            c_id = f"concept::{concept}"
            self.ensure_node(c_id, "concept", concept)
            self.add_edge(question_id, c_id, "Used_for")
            self.question_to_knowledge_nodes[question_id].add(c_id)

        for sub in sub_concept_names:
            s_id = f"sub_concept::{sub}"
            self.ensure_node(s_id, "sub_concept", sub)
            self.add_edge(question_id, s_id, "Part_of")
            self.question_to_knowledge_nodes[question_id].add(s_id)

        # Prefer explicit concept->sub-concept links.
        for concept in concept_names:
            c_id = f"concept::{concept}"
            for sub in sub_concept_names:
                s_id = f"sub_concept::{sub}"
                self.add_edge(s_id, c_id, "Hyponym_of")

        # If prerequisites are available, connect them.
        # IMPORTANT: prerequisite_concepts == sub_concepts in the problem bank.
        # Do NOT re-register them as "concept::" nodes  -  they are already
        # registered above as "sub_concept::" nodes.  Re-registering them as
        # concepts creates duplicate nodes (one sub_concept:: + one concept::
        # for every sub-concept), which doubles graph size and splits mastery
        # signals.  Only add prerequisite edges to nodes that already exist.
        prerequisites = list(problem.get("prerequisite_concepts", []) or [])
        for prereq in prerequisites:
            # Prefer the sub_concept:: node if it exists; fall back to concept::
            s_id = f"sub_concept::{prereq}"
            c_id = f"concept::{prereq}"
            if s_id in self.nodes:
                p_id = s_id
            elif c_id in self.nodes:
                p_id = c_id
            else:
                # Genuinely new prerequisite concept  -  register as concept node
                p_id = c_id
                self.ensure_node(p_id, "concept", prereq)
            for concept in concept_names:
                target_id = f"concept::{concept}"
                if p_id != target_id:
                    self.add_edge(p_id, target_id, "Prerequisite_of")

        return question_id

    def add_local_state(
        self,
        node_id: str,
        state: str,
        problem_id: Any,
        behavior: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> None:
        """Append a Good/Bad observation to the per-node history.

        ``weight`` controls how strongly this record contributes to
        ``summarize_node_state``.  Use 1.0 for full-strength signals
        (success, give-up) and a fraction (e.g. 0.5) for partial signals
        such as wrong-but-tried, where the student engaged with the
        concept but did not get the answer right.  This is what allows
        the LLM-visible ``mastery_score`` to decay after repeated wrong
        attempts instead of remaining stuck at the 0.65 default.
        """
        if state not in {"Good", "Bad"}:
            return
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 1.0
        if w <= 0.0:
            return
        self.local_state_library[node_id].append(
            {
                "state": state,
                "problem_id": problem_id,
                "behavior": behavior or {},
                "weight": w,
            }
        )

    def summarize_node_state(self, node_id: str) -> Dict[str, Any]:
        records = self.local_state_library.get(node_id, [])
        if not records:
            # Koedinger et al. (2023): prior for unseen concepts uses
            # population mean initial performance (~0.65)
            return {
                "global_state": "Unknown",
                "good_count": 0,
                "bad_count": 0,
                "mastery_score": 0.65,
            }
        good_w = 0.0
        bad_w = 0.0
        for r in records:
            w = float(r.get("weight", 1.0))
            if r.get("state") == "Good":
                good_w += w
            elif r.get("state") == "Bad":
                bad_w += w
        mastery = (good_w + 1.0) / (good_w + bad_w + 2.0)
        return {
            "global_state": "Good" if good_w >= bad_w else "Bad",
            "good_count": int(round(good_w)),
            "bad_count": int(round(bad_w)),
            "mastery_score": float(mastery),
        }

    def build_global_prototype(self) -> Dict[str, Dict[str, Any]]:
        prototype: Dict[str, Dict[str, Any]] = {}
        for node_id, node_data in self.nodes.items():
            if node_data.get("type") not in {"concept", "sub_concept"}:
                continue
            prototype[node_id] = {
                "label": node_data.get("label", node_id),
                "type": node_data.get("type"),
                **self.summarize_node_state(node_id),
            }
        return prototype

    def get_graph_snapshot(self) -> Dict[str, Any]:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_knowledge_nodes": sum(
                1
                for n in self.nodes.values()
                if n.get("type") in {"concept", "sub_concept"}
            ),
            "num_question_nodes": sum(
                1 for n in self.nodes.values() if n.get("type") == "question"
            ),
        }


class CognitiveStudentModel:
    """Simulates a student's cognitive processes and learning behavior"""

    # Type-specific base adjustment to give-up probability.
    # Positive = more likely to give up, negative = less likely.
    # Grounds: AMS motivation profiles and ITS engagement literature.
    _GIVEUP_TYPE_BASE: Dict[str, float] = {
        'analytical':        -0.12,
        'fast_learner':      -0.08,
        'confident_learner': -0.08,
        'average':            0.00,
        'intuitive':          0.04,
        'struggling_learner': 0.06,
        'anxious_learner':    0.10,
    }

    def __init__(self,
                 student_id: str,
                 initial_knowledge_level: float = 0.3,
                 learning_rate: float = 0.1,
                 attention_span: int = 10,
                 motivation_level: float = 0.7,
                 student_type: str = "average",
                 concept_dim: int = 58,
                 decision_agent: Optional[Any] = None,
                 solver_agent: Optional[Any] = None,
                 history_size: int = 50):

        self.student_id = student_id
        self.student_type = student_type
        self.concept_dim = concept_dim
        self.decision_agent = decision_agent
        self.solver_agent = solver_agent
        self.history_size = history_size
        self.knowledge_graph = StudentKnowledgeGraph()
        self.top_p_concepts = 5
        # When using a local Ollama model, reduce beam search to avoid
        # excessive inference time (each call ~30s on local GPU).
        import os as _os
        _use_local = _os.getenv("USE_LOCAL", "").strip().lower() in {"1", "true", "yes"}
        self.refinement_rounds = 1 if _use_local else 3
        self.refinement_beam_size = 1 if _use_local else 2
        self.refinement_threshold = 0.9

        # Core cognitive parameters
        # Koedinger et al. (2023): mean initial performance ≈ 0.65, range 0.55–0.75
        # KC-wise values drawn ±0.15 around student-level mean
        k_lo = max(0.05, initial_knowledge_level - 0.15)
        k_hi = min(0.98, initial_knowledge_level + 0.15)
        self.knowledge_state = np.random.uniform(k_lo, k_hi, self.concept_dim)
        self.base_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.attention_span = attention_span
        self.base_motivation = motivation_level
        self.motivation_level = motivation_level

        # Dynamic state variables
        self.current_session_problems = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.recent_performance = deque(maxlen=5)
        self.recent_answer_correctness = deque(maxlen=5)
        # Fatigue: SMFS baseline M=2.79/5 → session-start ~0.02–0.12
        self.fatigue_level = np.random.uniform(0.02, 0.12)
        self.confidence_level = 0.5
        self.last_answer_was_graded = False
        self.last_answer_correct: Optional[bool] = None
        self.last_help_requested = False
        self.last_needs_more_training = False
        self.last_gave_up = False
        self.last_confidence = 0.5

        # Learning style parameters
        self.prefers_visual = np.random.uniform(0.3, 0.8)   # Not using this, thought for the future refrence
        self.prefers_step_by_step = np.random.uniform(0.4, 0.9)
        self.challenge_tolerance = np.random.uniform(0.2, 0.8) # Not using this, thought for the future refrence
        self.help_seeking_tendency = np.random.uniform(0.1, 0.7)
        self.careless_error_rate = np.random.uniform(0.05, 0.20)
        self.overconfidence_bias = 0.0
        # Smallwood & Schooler (2013): baseline mind-wandering ~30%
        self.mind_wandering_base = 0.30

        # Performance tracking
        self.total_problems_attempted = 0
        self.total_problems_solved = 0

        # Concept-level learning history for cognitive prototype modeling
        self.concept_attempts = np.zeros(self.concept_dim, dtype=np.float32)
        self.concept_successes = np.zeros(self.concept_dim, dtype=np.float32)
        self.learning_history: Deque[Dict[str, Any]] = deque(maxlen=self.history_size)
        self.latest_cognitive_prototype: Optional[Dict[str, Any]] = None
        # External hint set by the tutoring environment when the student failed
        # the previous step. Consumed at the start of attempt_problem().
        self._pending_remediation_hint: Optional[str] = None

    def reset_session(self):
        """Reset per-session state without wiping long-term knowledge."""
        self.current_session_problems = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.recent_performance = deque(maxlen=5)
        self.recent_answer_correctness = deque(maxlen=5)
        self.fatigue_level = np.random.uniform(0.02, 0.12)
        self.confidence_level = 0.5
        self.motivation_level = self.base_motivation
        self.last_answer_was_graded = False
        self.last_answer_correct = None
        self.last_help_requested = False
        self.last_needs_more_training = False
        self.last_gave_up = False
        self.last_confidence = 0.5
        self._pending_remediation_hint = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        """Convert probability to log-odds, clamping to avoid ±inf."""
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Convert log-odds to probability."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def attempt_problem(self, problem: Dict) -> Dict:
        """Simulate student's attempt at solving a problem"""
        self.total_problems_attempted += 1

        # Consume any externally-set remediation hint (set by tutoring env).
        # Must be cleared before any early-return path to avoid stale state.
        _external_hint: Optional[str] = self._pending_remediation_hint
        self._pending_remediation_hint = None

        # Build graph-aware cognitive prototype for the current problem context
        prototype = self._build_cognitive_prototype(problem)
        # Statistical prior: success probability and error types used as context
        # for the DecisionAgent prompt  -  not a decision path, purely informational.
        _hint_available = bool(_external_hint)
        behavior = self._predict_behavior(problem, prototype, hint_available=_hint_available)
        epistemic_state_spec = self._build_epistemic_state_spec(
            problem, behavior, prototype, external_hint=_external_hint
        )
        success_probability = behavior["success_probability"]

        # Log the full problem being attempted
        _k_mean = float(np.mean(self.knowledge_state))
        import json as _json
        logger.debug(
            "  [Problem] FULL:\n%s",
            _json.dumps(
                {k: v for k, v in problem.items() if k != "concept_vector"},
                indent=2, default=str,
            ),
        )

        # Log the complete knowledge graph (all nodes with mastery scores)
        _kg_full = self.knowledge_graph.build_global_prototype()
        logger.debug(
            "  [KnowledgeGraph] FULL (%d nodes):\n%s",
            len(_kg_full),
            _json.dumps(_kg_full, indent=2, default=str),
        )

        # Log cognitive scalars and behavior prior
        logger.debug(
            "  [CognitiveState] student=%s  K_mean=%.4f  motivation=%.3f"
            "  fatigue=%.3f  attention_span=%s  consec_failures=%s",
            self.student_type, _k_mean, self.motivation_level,
            self.fatigue_level,
            getattr(self, "attention_span", "?"),
            getattr(self, "consecutive_failures", "?"),
        )
        logger.debug(
            "  [BehaviorPrior] FULL:\n%s",
            _json.dumps(behavior, indent=2, default=str),
        )

        # Dual-agent answer generation with beam-search refinement.
        answer_payload = self._simulate_answer(
            problem,
            behavior,
            prototype,
            epistemic_state_spec,
            success_probability,
        )

        # Determine give-up/help behavior.
        #
        # Contract: when the dual agents commit to an attempt (DecisionAgent
        # said can_solve=true AND SolverAgent did not set gave_up=true), we
        # respect that commitment  -  neither the heuristic predicted_outcome
        # nor the stochastic _check_if_gave_up roll may flip the outcome to
        # give-up.  This preserves the per-step LLM verdict the dual-agent
        # architecture was designed to deliver.
        #
        # The heuristic and stochastic checks remain active when the LLM
        # itself signalled give-up, returned ``can_solve=False``, or
        # produced no usable answer  -  i.e. when the LLM is also uncertain.
        # In those cases the heuristic acts as a tiebreaker over the
        # student's affective state (low motivation, high fatigue,
        # repeated failures) rather than overriding a clear LLM decision.
        llm_says_solve = bool(answer_payload.get("can_solve", False))
        llm_explicit_giveup = bool(answer_payload.get("gave_up", False))
        if llm_says_solve and not llm_explicit_giveup:
            gave_up = False
        else:
            gave_up = bool(
                llm_explicit_giveup
                or behavior["predicted_outcome"] == "give_up"
                or self._check_if_gave_up(
                    problem,
                    success_probability,
                    hint_available=_hint_available,
                )
            )

        # Grade answer against ground truth.  The dual-agent always produces
        # a final_answer (or None when can_solve=false), so grading should
        # always succeed when the problem has an answer key.
        graded_success = self._grade_answer(problem, answer_payload.get("final_answer"))
        if gave_up:
            success = False
            was_graded = graded_success is not None
        elif graded_success is not None:
            success = graded_success
            was_graded = True
        else:
            # Problem has no answer key  -  use DecisionAgent's can_solve verdict
            success = bool(answer_payload.get("can_solve", False))
            was_graded = False

        # Check if student requests help
        help_requested = bool(
            answer_payload.get("help_requested", False)
            or self._check_help_request(problem, success, success_probability)
        )

        # Calculate time spent
        time_spent = self._calculate_time_spent(problem, success, success_probability)

        # NOTE on dict construction order: ``answer_payload`` is spread FIRST
        # so the explicit keys below override LLM-supplied values.  Without
        # this ordering, ``**answer_payload`` was overwriting our computed
        # ``gave_up`` (the result of the LLM-commit-respecting logic above)
        # and ``help_requested`` with the raw LLM values, silently undoing
        # those decisions.  This was a pre-existing dict-merge bug exposed
        # once the give-up logic stopped agreeing with answer_payload by
        # coincidence.
        result = {
            **answer_payload,
            'success': success and not gave_up,
            'time_spent': time_spent,
            'confidence': success_probability,
            'gave_up': gave_up,
            'help_requested': help_requested,
            'problem_id': problem['id'],
            'behavior': behavior,
            'cognitive_prototype': prototype,
            'epistemic_state_spec': epistemic_state_spec,
            'answer_was_graded': was_graded,
            'answer_correct': bool(graded_success) if was_graded else None,
            'expected_answer': problem.get('answer'),
        }

        if not result["success"]:
            result["needs_more_training"] = bool(result.get("needs_more_training", True))
            result["practice_targets"] = self._normalize_practice_targets(
                result.get("practice_targets"),
                prototype.get("mapped_top_nodes", []),
            )
            if not result.get("hint_text"):
                result["hint_text"] = self._build_default_hint(prototype)
            # Wrong attempt (not give-up) gets an extra penalty signal.
            result["mistake_penalty"] = 1.0 if not gave_up else 0.0
        else:
            result["needs_more_training"] = bool(result.get("needs_more_training", False))
            result["practice_targets"] = self._normalize_practice_targets(
                result.get("practice_targets"),
                [],
            )
            result["mistake_penalty"] = 0.0

        self._update_state_after_attempt(problem, result)

        _k_mean_after = float(np.mean(self.knowledge_state))
        logger.debug(
            "  [KnowledgeUpdate] ΔK_mean=%+.5f  (%.4f → %.4f)  llm_failed=%s",
            _k_mean_after - _k_mean,
            _k_mean,
            _k_mean_after,
            result.get("llm_failed", False),
        )
        # Full result dict (strips numpy arrays for JSON serialisation)
        logger.debug(
            "  [AttemptResult] FULL:\n%s",
            _json.dumps(
                {k: v for k, v in result.items()
                 if k not in ("cognitive_prototype", "epistemic_state_spec", "behavior")},
                indent=2, default=str,
            ),
        )
        return result

    def _estimate_mastery_scores(self) -> np.ndarray:
        """Estimate per-concept mastery by blending knowledge and historical success."""
        attempts = self.concept_attempts
        successes = self.concept_successes
        history_mastery = (successes + 1.0) / (attempts + 2.0)
        return (0.6 * self.knowledge_state) + (0.4 * history_mastery)

    def _extract_active_concepts(self, problem_vector: np.ndarray) -> List[int]:
        """Return indices of active concepts for the problem."""
        return list(np.where(problem_vector > 0.0)[0])

    def _build_cognitive_prototype(self, problem: Dict) -> Dict[str, Any]:
        """
        Build a graph-aware cognitive prototype.

        Mirrors the paper's flow in a lightweight form:
        - map task to top-p relevant knowledge nodes
        - infer local/global cognitive state from node history
        - retrieve the most relevant historical record
        """
        problem_vector = np.array(problem['concept_vector'])
        active_indices = self._extract_active_concepts(problem_vector)
        concept_names, sub_concept_names = self._extract_problem_knowledge_units(
            problem, active_indices
        )

        # Ensure graph contains this problem's structure so node lookup is stable.
        question_node = self.knowledge_graph.register_problem(
            problem,
            concept_names=concept_names,
            sub_concept_names=sub_concept_names,
        )

        global_prototype = self.knowledge_graph.build_global_prototype()
        mastery_scores = self._estimate_mastery_scores()

        # Task-to-node mapping with top-p selection.
        scored_nodes = self._score_knowledge_nodes_for_task(
            problem=problem,
            concept_names=concept_names,
            sub_concept_names=sub_concept_names,
            global_prototype=global_prototype,
        )
        top_nodes = [node_id for node_id, _ in scored_nodes[: self.top_p_concepts]]

        # Derive mastery buckets from active vector indices for compatibility.
        mastered = [i for i in active_indices if mastery_scores[i] >= 0.7]
        weak = [i for i in active_indices if 0.4 <= mastery_scores[i] < 0.7]
        unknown = [i for i in active_indices if mastery_scores[i] < 0.4]

        mastery_ratio = 0.0
        if active_indices:
            weights = problem_vector[active_indices]
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                mastery_ratio = float(np.sum(weights * mastery_scores[active_indices]) / weight_sum)

        # Blend graph-level state over mapped nodes.
        if top_nodes:
            graph_mastery = np.mean(
                [global_prototype.get(node, {}).get("mastery_score", 0.65) for node in top_nodes]
            )
            mastery_ratio = float((mastery_ratio * 0.6) + (graph_mastery * 0.4))

        similar_record = self._retrieve_similar_record(top_nodes, problem_vector)

        prototype = {
            'active_indices': active_indices,
            'question_node': question_node,
            'concept_names': concept_names,
            'sub_concept_names': sub_concept_names,
            'mapped_top_nodes': top_nodes,
            'mapped_node_scores': scored_nodes[: self.top_p_concepts],
            'global_cognitive_states': {
                node_id: global_prototype.get(node_id, {})
                for node_id in top_nodes
            },
            'mastered_indices': mastered,
            'weak_indices': weak,
            'unknown_indices': unknown,
            'mastery_ratio': mastery_ratio,
            'similar_record': similar_record,
        }

        self.latest_cognitive_prototype = prototype
        return prototype

    def _retrieve_similar_record(
        self,
        mapped_top_nodes: List[str],
        problem_vector: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve most relevant historical attempt by top-p node overlap."""
        if not self.learning_history:
            return None

        best_overlap = -1
        best_similarity = -1.0
        best_record = None

        for record in self.learning_history:
            past_nodes = set(record.get("knowledge_nodes", []))
            overlap = len(set(mapped_top_nodes) & past_nodes)
            # Tie-break by vector cosine similarity when available.
            past_vector = record.get("concept_vector")
            similarity = 0.0
            if past_vector is not None:
                denom = np.linalg.norm(problem_vector) * np.linalg.norm(past_vector)
                if denom > 0:
                    similarity = float(np.dot(problem_vector, past_vector) / denom)
            if overlap > best_overlap or (overlap == best_overlap and similarity > best_similarity):
                best_overlap = overlap
                best_similarity = similarity
                best_record = record

        # Only return a record when there is genuine concept overlap.
        # A zero-overlap record would be topically unrelated and its hint
        # would mislead the student model.
        if best_record is None or best_overlap == 0:
            return None

        return {
            'problem_id': best_record.get('problem_id'),
            'success': best_record.get('success'),
            'difficulty_score': best_record.get('difficulty_score'),
            'overlap': best_overlap,
            'similarity': best_similarity,
            'predicted_outcome': best_record.get('predicted_outcome'),
            # Carry forward any hint generated for a failed similar attempt
            # so it can be surfaced as prior scaffolding guidance.
            'hint_text': best_record.get('hint_text', '') or '',
        }

    def _extract_problem_knowledge_units(
        self,
        problem: Dict[str, Any],
        active_indices: List[int],
    ) -> tuple[List[str], List[str]]:
        concept_names = list(problem.get("concepts", []) or [])
        sub_concept_names = list(problem.get("sub_concepts", []) or [])

        if not concept_names:
            concept_names = [f"concept_idx_{idx}" for idx in active_indices]
        if not sub_concept_names:
            # If sub-concepts are absent, we still create fine-grained slots from active indices.
            sub_concept_names = [f"subconcept_idx_{idx}" for idx in active_indices]

        # Deduplicate while preserving order.
        concept_names = list(dict.fromkeys(str(c) for c in concept_names))
        sub_concept_names = list(dict.fromkeys(str(s) for s in sub_concept_names))
        return concept_names, sub_concept_names

    def _task_text(self, problem: Dict[str, Any]) -> str:
        parts = [
            str(problem.get("question", "")),
            str(problem.get("solution", "")),
            " ".join(problem.get("concepts", []) or []),
            " ".join(problem.get("sub_concepts", []) or []),
        ]
        return " ".join(p for p in parts if p).lower()

    def _token_overlap_score(self, source: str, target: str) -> float:
        s_tokens = set(source.lower().split())
        t_tokens = set(target.lower().split())
        if not s_tokens or not t_tokens:
            return 0.0
        intersection = len(s_tokens & t_tokens)
        return float(intersection / max(1, len(s_tokens | t_tokens)))

    def _score_knowledge_nodes_for_task(
        self,
        problem: Dict[str, Any],
        concept_names: List[str],
        sub_concept_names: List[str],
        global_prototype: Dict[str, Dict[str, Any]],
    ) -> List[tuple[str, float]]:
        task_text = self._task_text(problem)
        explicit_nodes = {f"concept::{c}" for c in concept_names} | {
            f"sub_concept::{s}" for s in sub_concept_names
        }

        scored: List[tuple[str, float]] = []
        for node_id, node_data in self.knowledge_graph.nodes.items():
            node_type = node_data.get("type")
            if node_type not in {"concept", "sub_concept"}:
                continue
            summary = global_prototype.get(node_id, {})
            mastery = float(summary.get("mastery_score", 0.5))
            lexical = self._token_overlap_score(task_text, str(node_data.get("label", "")))
            explicit_bonus = 1.0 if node_id in explicit_nodes else 0.0
            score = (0.45 * mastery) + (0.25 * lexical) + (0.30 * explicit_bonus)
            scored.append((node_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _normalize_practice_targets(
        self,
        raw_targets: Any,
        fallback: List[str],
    ) -> List[str]:
        if not isinstance(raw_targets, list):
            raw_targets = []
        normalized = [str(x).strip() for x in raw_targets if str(x).strip()]
        if not normalized:
            normalized = [str(x).strip() for x in fallback if str(x).strip()]
        return normalized[:5]

    def _build_default_hint(self, prototype: Dict[str, Any]) -> str:
        concept_names = list(prototype.get("concept_names", []) or [])
        sub_concept_names = list(prototype.get("sub_concept_names", []) or [])
        if sub_concept_names:
            return f"Let's review this sub-concept first: {sub_concept_names[0]}."
        if concept_names:
            return f"Let's review this concept first: {concept_names[0]}."
        return "Let's review a similar prerequisite question first."

    # Observed difficulty range in the GSM8K problem bank (set once).
    _DIFF_LO = 0.41
    _DIFF_HI = 0.89

    def _predict_behavior(self, problem: Dict, prototype: Dict[str, Any], hint_available: bool = False) -> Dict[str, Any]:
        """Predict behavior from cognitive prototype and affective state."""
        # Normalize difficulty_score from the observed [0.41, 0.89] range to [0, 1]
        # so the formula works correctly across the full spectrum.
        raw_diff = problem['difficulty_score']
        norm_diff = (raw_diff - self._DIFF_LO) / max(self._DIFF_HI - self._DIFF_LO, 1e-6)
        norm_diff = max(0.0, min(1.0, norm_diff))
        # norm_diff ∈ [0,1] → factor: easy(0.0)→1.15, medium(0.5)→0.75, hard(1.0)→0.35
        difficulty_factor = max(0.15, 1.15 - 0.80 * norm_diff)
        fatigue_penalty = self.fatigue_level * 0.3
        # AMS-derived motivation: population mean ~0.68, center bonus at 0.65
        motivation_bonus = (self.motivation_level - 0.65) * 0.25

        # Smallwood & Schooler (2013): baseline mind-wandering ~30%, modulated
        # upward by fatigue/boredom and downward by motivation/interest
        mind_wandering_prob = self.mind_wandering_base + (self.fatigue_level * 0.15) - (self.motivation_level * 0.10)
        mind_wandering_prob = max(0.05, min(0.60, mind_wandering_prob))
        attention_penalty = mind_wandering_prob * 0.15

        mapped_states = prototype.get("global_cognitive_states", {})
        if mapped_states:
            mapped_mastery = float(
                np.mean([v.get("mastery_score", 0.65) for v in mapped_states.values()])
            )
        else:
            mapped_mastery = prototype["mastery_ratio"]

        history_bias = 0.0
        similar_record = prototype.get("similar_record")
        if similar_record is not None:
            history_bias = 0.08 if similar_record.get("success") else -0.08

        base_probability = mapped_mastery * difficulty_factor
        # Hint bonus: prior scaffolding guidance raises effective success probability.
        # Grounded in ITS literature: worked examples / hints raise P(correct) by
        # 10–18 percentage points (VanLehn, 2011; Aleven & Koedinger, 2000).
        hint_bonus = 0.12 if hint_available else 0.0
        adjusted_probability = base_probability + motivation_bonus - fatigue_penalty - attention_penalty + hint_bonus
        adjusted_probability += history_bias
        success_probability = max(0.05, min(0.95, adjusted_probability))

        # Give-up threshold: raised from 0.25 to 0.40 to match AMS-derived
        # motivation ranges (population min ~0.38 for struggling learners)
        if self.motivation_level < 0.40 and self.fatigue_level > 0.6:
            outcome = 'give_up'
        elif success_probability > 0.75:
            outcome = 'success'
        elif success_probability > 0.45:
            outcome = 'partial' if prototype['unknown_indices'] else 'success'
        else:
            outcome = 'wrong'

        error_types = self._infer_error_types(problem, prototype, success_probability)

        return {
            'success_probability': success_probability,
            'predicted_outcome': outcome,
            'error_types': error_types,
            'mastery_ratio': prototype['mastery_ratio'],
        }

    def _infer_error_types(
        self,
        problem: Dict,
        prototype: Dict[str, Any],
        success_probability: float,
    ) -> List[str]:
        """Infer likely error types from the cognitive prototype."""
        error_types = []

        if prototype['unknown_indices']:
            error_types.append('missing_concept')
        if len(prototype['unknown_indices']) >= max(1, len(prototype['active_indices']) // 2):
            error_types.append('misconception')
        if self.fatigue_level > 0.6:
            error_types.append('careless_error')
        if problem['difficulty_score'] > 0.7:
            error_types.append('procedure_error')
        if prototype['mastery_ratio'] > 0.6 and success_probability < 0.5:
            error_types.append('calculation_error')

        if not error_types:
            error_types.append('unspecified_error')

        return error_types

    def _build_epistemic_state_spec(
        self,
        problem: Dict,
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
        external_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build an ESS-style specification that can condition an external generator.

        The `prior_remediation_hint` field surfaces any hint generated for a
        previously failed attempt on a similar problem, so the DecisionAgent
        and SolverAgent can take it into account (e.g. apply a scaffolding
        strategy that was already suggested once).
        """
        recent_success_rate = (
            sum(self.recent_performance) / len(self.recent_performance)
            if self.recent_performance
            else 0.5
        )
        # Priority: explicit external hint (from tutoring env) > hint stored
        # on the most similar historical failure.
        prior_hint: Optional[str] = external_hint or None
        if not prior_hint:
            similar_record = prototype.get('similar_record')
            if similar_record and not similar_record.get('success', True):
                prior_hint = similar_record.get('hint_text') or None

        return {
            'problem_id': problem.get('id'),
            'problem_difficulty': problem.get('difficulty_score'),
            'concept_names': prototype.get('concept_names', []),
            'sub_concept_names': prototype.get('sub_concept_names', []),
            'known_concepts': prototype.get('mastered_indices', []),
            'weak_concepts': prototype.get('weak_indices', []),
            'unknown_concepts': prototype.get('unknown_indices', []),
            'mapped_top_nodes': prototype.get('mapped_top_nodes', []),
            'mapped_node_scores': prototype.get('mapped_node_scores', []),
            'graph_states': prototype.get('global_cognitive_states', {}),
            'predicted_outcome': behavior.get('predicted_outcome'),
            'error_types': behavior.get('error_types', []),
            'motivation_level': self.motivation_level,
            'fatigue_level': self.fatigue_level,
            'recent_success_rate': recent_success_rate,
            'consecutive_failures': self.consecutive_failures,
            'help_seeking_tendency': self.help_seeking_tendency,
            'prior_remediation_hint': prior_hint,
        }

    def _build_decision_context(
        self,
        problem: Dict[str, Any],
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
        epistemic_state_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assemble rich context for the DecisionAgent.

        Includes the full knowledge graph prototype (all concept mastery scores),
        recent attempt history, and current cognitive state.
        """
        # Serialize recent history to simple dicts (skip numpy concept_vector)
        recent_history = []
        for record in list(self.learning_history)[-10:]:
            recent_history.append({
                "problem_id": record.get("problem_id"),
                "question_text": record.get("question_text", ""),
                "correct_answer": record.get("correct_answer"),
                "student_final_answer": record.get("student_final_answer"),
                "student_answer_text": record.get("student_answer_text", ""),
                "answer_correct": record.get("answer_correct"),
                "success": record.get("success"),
                "gave_up": record.get("gave_up", False),
                "help_requested": record.get("help_requested", False),
                "difficulty_score": float(record.get("difficulty_score", 0.5))
                if record.get("difficulty_score") is not None else None,
                "predicted_outcome": record.get("predicted_outcome"),
                "error_types": list(record.get("error_types", [])),
                "knowledge_nodes": list(record.get("knowledge_nodes", [])),
                # Diagnostic signals from failed attempts  -  tells the LLM
                # exactly which concepts were weak and what guidance was given.
                "hint_text": record.get("hint_text", ""),
                "practice_targets": list(record.get("practice_targets", [])),
            })

        # Include question nodes from the knowledge graph so the DecisionAgent
        # can see which exact problems the student has encountered before.
        past_questions = []
        for node_id, node_data in self.knowledge_graph.nodes.items():
            if node_data.get("type") == "question":
                linked_concepts = list(
                    self.knowledge_graph.question_to_knowledge_nodes.get(node_id, set())
                )
                past_questions.append({
                    "node_id": node_id,
                    "question_text": str(node_data.get("label", ""))[:200],
                    "linked_concepts": linked_concepts,
                })

        return {
            "knowledge_graph_summary": self.knowledge_graph.build_global_prototype(),
            "past_questions_in_graph": past_questions,
            "recent_history": recent_history,
            "graph_snapshot": self.knowledge_graph.get_graph_snapshot(),
            "mapped_top_nodes": prototype.get("mapped_top_nodes", []),
            "graph_states": prototype.get("global_cognitive_states", {}),
            "student_state": self.get_current_state(),
            "epistemic_state_spec": epistemic_state_spec,
            "heuristic_prediction": behavior,
            # Per-instance persona traits sampled at student creation
            # (prefers_step_by_step, careless_error_rate, overconfidence_bias,
            # help_seeking_tendency, challenge_tolerance).  Without this the
            # DecisionAgent only knows the generic system-prompt persona text
            # and cannot distinguish two students of the same type with
            # different sampled trait values.
            "student_persona": self._get_student_persona(),
            # Surface the prior hint at the top level so the DecisionAgent
            # prompt template can easily reference it without digging into the ESS.
            "prior_remediation_hint": epistemic_state_spec.get("prior_remediation_hint"),
        }

    def _get_student_persona(self) -> Dict[str, Any]:
        """Return persona dict for DecisionAgent and SolverAgent conditioning.

        Includes every per-instance trait that influences the simulation,
        so two students of the same type with different sampled values are
        distinguishable to the LLM.  ``prefers_visual`` is omitted because
        the math-tutoring task has no visual modality.
        """
        return {
            "student_type": self.student_type,
            "prefers_step_by_step": float(self.prefers_step_by_step),
            "help_seeking_tendency": float(self.help_seeking_tendency),
            "careless_error_rate": float(self.careless_error_rate),
            "overconfidence_bias": float(self.overconfidence_bias),
            "challenge_tolerance": float(self.challenge_tolerance),
            "mind_wandering_base": float(self.mind_wandering_base),
            "attention_span": int(self.attention_span),
            "learning_rate": float(self.base_learning_rate),
        }

    def _simulate_answer(
        self,
        problem: Dict,
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
        epistemic_state_spec: Dict[str, Any],
        success_probability: float,
    ) -> Dict[str, Any]:
        """
        Simulate answer generation via the dual-agent architecture.

        1. DecisionAgent decides can_solve based on full knowledge graph state.
        2. SolverAgent generates the answer conditioned on that decision,
           with beam-search refinement (L rounds, B candidates per round).
        """
        if self.decision_agent is None or self.solver_agent is None:
            raise RuntimeError(
                "Both decision_agent and solver_agent are required. "
                "Set USE_LOCAL=true for Ollama, or set OPENAI_API_KEY for OpenAI."
            )

        decision_context = self._build_decision_context(
            problem, behavior, prototype, epistemic_state_spec
        )
        decision = self.decision_agent(problem, behavior, prototype, decision_context)
        student_persona = self._get_student_persona()

        # Initial candidate from solver
        best_candidate = self.solver_agent(
            problem=problem,
            decision=decision,
            student_persona=student_persona,
            previous_candidate=None,
        )
        best_score = self._score_solution_alignment(problem, behavior, best_candidate)

        # If the first solver call already exhausted retries and fell back,
        # skip beam refinement to avoid repeated failing API calls.
        if best_candidate.get("llm_failed"):
            best_candidate["alignment_score"] = best_score
            if "predicted_outcome" not in best_candidate:
                best_candidate["predicted_outcome"] = decision.get(
                    "predicted_outcome", behavior.get("predicted_outcome")
                )
            if "error_types" not in best_candidate:
                best_candidate["error_types"] = decision.get(
                    "error_types", behavior.get("error_types", [])
                )
            return best_candidate

        # Beam-search refinement rounds
        for _ in range(self.refinement_rounds):
            if best_score >= self.refinement_threshold:
                break
            candidates = []
            for _beam in range(self.refinement_beam_size):
                candidate = self.solver_agent(
                    problem=problem,
                    decision=decision,
                    student_persona=student_persona,
                    previous_candidate=best_candidate,
                )
                if candidate.get("llm_failed"):
                    continue
                score = self._score_solution_alignment(problem, behavior, candidate)
                candidates.append((score, candidate))
            if not candidates:
                break
            round_best_score, round_best_candidate = max(candidates, key=lambda x: x[0])
            if round_best_score >= best_score:
                best_score = round_best_score
                best_candidate = round_best_candidate

        # Merge decision-agent fields into candidate so downstream code sees them
        best_candidate["alignment_score"] = best_score
        if "predicted_outcome" not in best_candidate:
            best_candidate["predicted_outcome"] = decision.get(
                "predicted_outcome", behavior.get("predicted_outcome")
            )
        if "error_types" not in best_candidate:
            best_candidate["error_types"] = decision.get(
                "error_types", behavior.get("error_types", [])
            )
        return best_candidate

    def _score_solution_alignment(
        self,
        problem: Dict[str, Any],
        behavior: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> float:
        """
        Score how well a candidate matches predicted student cognition.
        """
        predicted_outcome = behavior.get("predicted_outcome", "wrong")
        final_answer = candidate.get("final_answer")
        graded = self._grade_answer(problem, final_answer)
        answer_text = str(candidate.get("answer_text", "")).lower()

        score = 0.0
        if predicted_outcome == "give_up":
            # Only the explicit ``gave_up`` flag counts as give-up evidence.
            # Phrases like "I'm not sure" and "I don't know" are persona-
            # canonical for struggling and anxious students who are
            # ATTEMPTING in good faith  -  using them as give-up evidence
            # caused the beam-search refinement to upscore give-up
            # candidates whenever those personas spoke in their natural
            # voice, locking the simulation into give-up loops.
            if candidate.get("gave_up"):
                score += 0.75
            if candidate.get("help_requested"):
                score += 0.15
        elif predicted_outcome == "success":
            if graded is True:
                score += 0.8
            elif graded is None and final_answer:
                score += 0.3
        elif predicted_outcome == "partial":
            if "stuck" in answer_text or candidate.get("help_requested"):
                score += 0.5
            if graded is False or graded is None:
                score += 0.25
        else:  # wrong
            if graded is False:
                score += 0.75
            elif graded is None:
                score += 0.35

        if behavior.get("error_types"):
            if predicted_outcome in {"partial", "wrong"}:
                score += 0.1

        return float(min(1.0, score))

    def _grade_answer(self, problem: Dict[str, Any], final_answer: Any) -> Optional[bool]:
        """
        Grade correctness using FINAL ANSWER ONLY (never reasoning/process text).
        Returns None only when no ground-truth target answer exists.
        """
        target = problem.get("answer")
        if target is None:
            return None
        if final_answer is None:
            return False

        target_num = self._parse_numeric_answer(target)
        final_num = self._parse_numeric_answer(final_answer)
        if target_num is not None and final_num is not None:
            return bool(abs(target_num - final_num) <= 1e-6)

        target_text = str(target).strip().lower()
        final_text = str(final_answer).strip().lower()
        if not target_text:
            return None
        if not final_text:
            return False
        return target_text == final_text

    def _parse_numeric_answer(self, answer: Any) -> Optional[float]:
        """Parse numeric answers including simple fractions."""
        if answer is None:
            return None
        text = str(answer).strip()
        if not text:
            return None
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2:
                try:
                    num = float(parts[0].strip())
                    den = float(parts[1].strip())
                    if den != 0:
                        return num / den
                except ValueError:
                    return None
        try:
            return float(text)
        except ValueError:
            return None

    def _mutate_numeric_answer(self, answer: Any, error_types: List[str]) -> Optional[str]:
        """Create a plausible incorrect numeric answer based on error types."""
        value = self._parse_numeric_answer(answer)
        if value is None:
            return None

        candidates = []
        if 'careless_error' in error_types:
            candidates.extend([value + 1, value - 1, -value])
        if 'procedure_error' in error_types:
            candidates.extend([value * 2, value / 2 if value != 0 else 0])
        if 'calculation_error' in error_types:
            candidates.extend([value + 2, value - 2])
        if 'missing_concept' in error_types:
            candidates.extend([value * 10, value / 10 if value != 0 else 0])

        candidates = [c for c in candidates if c != value]
        if not candidates:
            candidates = [value + 1, value - 1]

        mutated = random.choice(candidates)
        if abs(mutated - round(mutated)) < 1e-6:
            return str(int(round(mutated)))
        return f"{mutated:.4g}"

    def _calculate_time_spent(self, problem: Dict, success: bool, confidence: float) -> int:
        """Calculate time spent on problem in seconds"""
        base_time = problem.get('estimated_time_minutes', 3) * 60

        if success:
            time_multiplier = 0.7 if confidence > 0.7 else 1.0
        else:
            time_multiplier = 1.5 if confidence > 0.5 else 1.2

        fatigue_multiplier = 1.0 + (self.fatigue_level * 0.3)
        total_time = base_time * time_multiplier * fatigue_multiplier
        total_time *= np.random.uniform(0.8, 1.2)

        return max(30, int(total_time))

    def _check_if_gave_up(
        self,
        problem: Dict,
        success_probability: float,
        hint_available: bool = False,
    ) -> bool:
        """Check if student gives up on the problem.

        Modelled on Arroyo et al. (2014) affect-aware ITS:
        - Type-specific base adjustment (engagement profile)
        - Hint reduces give-up probability (scaffolding effect)
        - Recent success streak lowers give-up probability
        - Threshold lowered to 0.40 (was 0.45) to avoid pre-dooming
          struggling_learner students whose motivation starts at 0.45
        """
        give_up_probability = 0.0

        # Type-specific base adjustment
        give_up_probability += self._GIVEUP_TYPE_BASE.get(self.student_type, 0.0)

        # Motivation below 0.40 (lowered from 0.45  -  gives struggling students
        # a fighting chance when motivation is in 0.40–0.44 range)
        if self.motivation_level < 0.40:
            give_up_probability += 0.30

        give_up_probability += self.fatigue_level * 0.20

        if success_probability < 0.20:
            give_up_probability += 0.40

        if self.consecutive_failures >= 3:
            give_up_probability += 0.25  # reduced from 0.30  -  failure counter no longer increments on give-up

        # Hint scaffolding: prior hint reduces give-up probability
        if hint_available:
            give_up_probability -= 0.15

        # Recent success streak: student on a roll is less likely to disengage
        if self.consecutive_successes >= 2:
            give_up_probability -= 0.10
        elif self.consecutive_successes >= 1:
            give_up_probability -= 0.05

        return np.random.random() < min(0.60, max(0.0, give_up_probability))

    def _check_help_request(self, problem: Dict, success: bool, confidence: float) -> bool:
        """Check if student requests help"""
        help_probability = self.help_seeking_tendency * 0.5

        if confidence < 0.3:
            help_probability += 0.3

        if len(self.recent_performance) >= 2:
            recent_success_rate = sum(self.recent_performance) / len(self.recent_performance)
            if recent_success_rate < 0.4:
                help_probability += 0.2

        return np.random.random() < min(0.8, help_probability)

    def _update_state_after_attempt(self, problem: Dict, result: Dict):
        """Update student's internal state after problem attempt"""
        success = result['success']
        gave_up = result['gave_up']

        problem_vector = np.array(problem['concept_vector'])
        active_indices = self._extract_active_concepts(problem_vector)
        prototype = result.get("cognitive_prototype", {})
        behavior = result.get("behavior", {})
        concept_names = list(prototype.get("concept_names", []) or [])
        sub_concept_names = list(prototype.get("sub_concept_names", []) or [])
        knowledge_nodes = [f"concept::{c}" for c in concept_names] + [
            f"sub_concept::{s}" for s in sub_concept_names
        ]

        # Ensure graph contains current problem and update local concept states.
        self.knowledge_graph.register_problem(
            problem,
            concept_names=concept_names,
            sub_concept_names=sub_concept_names,
        )
        # Graph state update:
        # - Success         → "Good" with weight 1.0 (full positive signal).
        # - Give-up         → "Bad"  with weight 1.0 (full negative  -  no learning).
        # - Wrong-but-tried → "Bad"  with weight 0.5 (half negative  -  engaged
        #                     but failed; matches the 30% knowledge_state bump
        #                     applied below).
        #
        # Without the wrong-but-tried branch the LLM-visible ``mastery_score``
        # for any concept the student keeps failing on stays pinned at the
        # 0.65 default forever (formula: (good+1)/(good+bad+2)).  That was
        # the root cause of the "no learning" symptom: the DecisionAgent saw
        # identical 0.65 masteries on every step regardless of past failures,
        # so it could not differentiate "first attempt" from "fifth failed
        # attempt".  Half weight keeps wrong attempts informative without
        # over-penalising  -  a single wrong attempt drops mastery from 0.65 to
        # ~0.40, five wrong attempts drop it to ~0.22.
        if success and not gave_up:
            for node_id in knowledge_nodes:
                self.knowledge_graph.add_local_state(
                    node_id=node_id,
                    state="Good",
                    problem_id=problem.get("id"),
                    behavior=behavior,
                )
        elif gave_up:
            for node_id in knowledge_nodes:
                self.knowledge_graph.add_local_state(
                    node_id=node_id,
                    state="Bad",
                    problem_id=problem.get("id"),
                    behavior=behavior,
                )
        else:
            for node_id in knowledge_nodes:
                self.knowledge_graph.add_local_state(
                    node_id=node_id,
                    state="Bad",
                    problem_id=problem.get("id"),
                    behavior=behavior,
                    weight=0.5,
                )

        # If model predicted unknown concepts, enforce bad local states for these indices.
        unknown_indices = set(prototype.get("unknown_indices", []) or [])
        for idx in unknown_indices:
            node_id = f"concept::concept_idx_{idx}"
            if node_id in self.knowledge_graph.nodes:
                self.knowledge_graph.add_local_state(
                    node_id=node_id,
                    state="Bad",
                    problem_id=problem.get("id"),
                    behavior=behavior,
                )

        if active_indices:
            self.concept_attempts[active_indices] += 1.0
            if success and not gave_up:
                self.concept_successes[active_indices] += 1.0
        self.learning_history.append({
            'problem_id': problem.get('id'),
            'question_text': str(problem.get('question', ''))[:200],  # cap to 200 chars
            'correct_answer': str(problem.get('answer', '')) if problem.get('answer') is not None else None,
            'student_final_answer': result.get('final_answer'),
            'student_answer_text': str(result.get('answer_text', ''))[:150],  # cap to 150 chars
            'answer_correct': result.get('answer_correct'),
            'concept_vector': problem_vector,
            'difficulty_score': problem.get('difficulty_score'),
            'success': success and not gave_up,
            'gave_up': gave_up,
            'help_requested': result.get('help_requested', False),
            'predicted_outcome': behavior.get("predicted_outcome"),
            'error_types': list(behavior.get("error_types", [])),
            'knowledge_nodes': knowledge_nodes,
            # Store the hint so future similar-problem attempts can retrieve it
            # via _retrieve_similar_record and surface it as prior guidance.
            'hint_text': str(result.get('hint_text', ''))[:300],
            # Store practice_targets so the LLM sees which specific concepts
            # were identified as weak  -  not just the hint text.
            'practice_targets': list(result.get('practice_targets', [])),
        })

        # Koedinger et al. (2023): update knowledge in logit space
        # μ ≈ 0.10 log-odds/opportunity (typical), ~7 opportunities to mastery
        if success and not gave_up:
            mu = self.learning_rate * (1.0 - self.fatigue_level)
            logit_k = self._logit(self.knowledge_state)
            logit_k += mu * problem_vector
            self.knowledge_state = self._sigmoid(logit_k)

            self.consecutive_failures = 0
            self.consecutive_successes += 1
            # Increased from +0.02: one success shouldn't be negligible.
            # VanLehn (2011): successful problem solving raises self-efficacy measurably.
            self.motivation_level = min(1.0, self.motivation_level + 0.05)
            # Consecutive success streak: re-engagement recovery
            if self.consecutive_successes >= 3:
                self.motivation_level = min(1.0, self.motivation_level + 0.05)
            # Fatigue partial recovery: successful easy problem restores attention.
            # Smallwood & Schooler (2013): interest/competence reduces mind-wandering.
            confidence = result.get("confidence", 0.5)
            if confidence > 0.65:
                self.fatigue_level = max(0.0, self.fatigue_level - 0.03)
            self.total_problems_solved += 1

        elif not gave_up:
            # Wrong answer: reduced learning (30% of full rate)
            mu = self.learning_rate * 0.3 * (1.0 - self.fatigue_level)
            logit_k = self._logit(self.knowledge_state)
            logit_k += mu * problem_vector
            self.knowledge_state = self._sigmoid(logit_k)

            self.consecutive_failures += 1
            self.consecutive_successes = 0

            if self.consecutive_failures > 2:
                self.motivation_level = max(0.1, self.motivation_level - 0.05)
        else:
            # Give-up: disengagement signal only  -  NOT a knowledge-failure signal.
            # Do NOT increment consecutive_failures; that counter tracks knowledge
            # struggles where the student tried but failed.  Give-up cascading
            # (each give-up making the next more likely) was a root cause of
            # 100% give-up rates in evaluation data.
            self.consecutive_successes = 0
            # Reduced from -0.10: still penalises disengagement but less catastrophic.
            self.motivation_level = max(0.1, self.motivation_level - 0.08)

        # Fatigue dynamics: accelerating after attention_span
        # Smallwood & Schooler (2013): lapse frequency increases from every
        # 10-18 min to every 3-4 min by end of session
        self.current_session_problems += 1
        if self.current_session_problems > self.attention_span:
            problems_over = self.current_session_problems - self.attention_span
            fatigue_increment = 0.05 + 0.02 * min(problems_over, 10)
            self.fatigue_level = min(1.0, self.fatigue_level + fatigue_increment)

        self.recent_performance.append(1 if success and not gave_up else 0)

        # Smoothly update ``confidence_level`` as a subjective self-efficacy
        # signal that reflects how the student feels right now: a blend of
        # recent performance, current motivation, and inverse fatigue.
        # Distinct from ``motivation_level`` (engagement/willingness) and
        # from ``heuristic_prior.success_probability`` (model belief).
        # Bandura (1997): self-efficacy is built incrementally from mastery
        # experiences, which is the recent-performance term here.
        recent_acc = (
            sum(self.recent_performance) / len(self.recent_performance)
            if self.recent_performance else 0.5
        )
        target_conf = (
            0.55 * float(recent_acc)
            + 0.25 * float(self.motivation_level)
            + 0.20 * float(1.0 - self.fatigue_level)
        )
        self.confidence_level = float(
            0.7 * self.confidence_level + 0.3 * target_conf
        )
        self.confidence_level = max(0.05, min(0.95, self.confidence_level))

        self.last_answer_was_graded = bool(result.get("answer_was_graded", False))
        self.last_answer_correct = (
            bool(result.get("answer_correct"))
            if self.last_answer_was_graded
            else None
        )
        self.last_help_requested = bool(result.get("help_requested", False))
        self.last_needs_more_training = bool(result.get("needs_more_training", False))
        self.last_gave_up = bool(result.get("gave_up", False))
        self.last_confidence = float(result.get("confidence", 0.5))
        if self.last_answer_was_graded:
            self.recent_answer_correctness.append(1 if self.last_answer_correct else 0)
        else:
            self.recent_answer_correctness.append(0.5)

    def get_current_state(self) -> Dict:
        """Get current student state for RL agent"""
        recent_success_rate = (sum(self.recent_performance) / len(self.recent_performance)
                              if self.recent_performance else 0.5)
        recent_answer_accuracy = (
            sum(self.recent_answer_correctness) / len(self.recent_answer_correctness)
            if self.recent_answer_correctness
            else 0.5
        )

        return {
            'knowledge_state': self.knowledge_state.copy(),
            'motivation_level': self.motivation_level,
            'fatigue_level': self.fatigue_level,
            'confidence_level': float(self.confidence_level),
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'recent_success_rate': recent_success_rate,
            'recent_answer_accuracy': recent_answer_accuracy,
            'current_session_problems': self.current_session_problems,
            'student_type': self.student_type,
            'last_answer_was_graded': self.last_answer_was_graded,
            'last_answer_correct': self.last_answer_correct,
            'last_help_requested': self.last_help_requested,
            'last_needs_more_training': self.last_needs_more_training,
            'last_gave_up': self.last_gave_up,
            'last_confidence': self.last_confidence,
            'knowledge_graph': self.knowledge_graph.get_graph_snapshot(),
        }

    def get_context_features(self, context_dim: int = 16) -> np.ndarray:
        """
        Build context features used by simulator and RL environment observations.
        """
        state = self.get_current_state()
        graph = state.get("knowledge_graph", {})
        last_answer_correct = state.get("last_answer_correct")
        if last_answer_correct is None:
            last_answer_correct_value = 0.5
        else:
            last_answer_correct_value = 1.0 if last_answer_correct else 0.0

        context_features = [
            min(1.0, max(0.0, float(state.get("motivation_level", 0.5)))),
            min(1.0, max(0.0, float(state.get("fatigue_level", 0.0)))),
            min(1.0, max(0.0, float(state.get("consecutive_failures", 0.0)) / 10.0)),
            min(1.0, max(0.0, float(state.get("recent_success_rate", 0.5)))),
            min(1.0, max(0.0, float(state.get("current_session_problems", 0.0)) / 20.0)),
            min(1.0, max(0.0, float(state.get("recent_answer_accuracy", 0.5)))),
            1.0 if bool(state.get("last_answer_was_graded", False)) else 0.0,
            min(1.0, max(0.0, last_answer_correct_value)),
            1.0 if bool(state.get("last_help_requested", False)) else 0.0,
            1.0 if bool(state.get("last_needs_more_training", False)) else 0.0,
            1.0 if bool(state.get("last_gave_up", False)) else 0.0,
            min(1.0, max(0.0, float(state.get("last_confidence", 0.5)))),
            min(1.0, max(0.0, float(graph.get("num_nodes", 0.0)) / 300.0)),
            min(1.0, max(0.0, float(graph.get("num_edges", 0.0)) / 600.0)),
            min(1.0, max(0.0, float(graph.get("num_knowledge_nodes", 0.0)) / 300.0)),
            min(1.0, max(0.0, float(state.get("consecutive_successes", 0.0)) / 10.0)),
        ]

        while len(context_features) < context_dim:
            context_features.append(0.0)

        return np.array(context_features[:context_dim], dtype=np.float32)

    def get_full_knowledge_graph(self) -> Dict[str, Any]:
        """Return full graph structure and local/global cognitive state summaries."""
        return {
            "nodes": self.knowledge_graph.nodes,
            "edges": {
                f"{src}->{dst}": rel
                for (src, dst), rel in self.knowledge_graph.edges.items()
            },
            "global_prototype": self.knowledge_graph.build_global_prototype(),
            "graph_snapshot": self.knowledge_graph.get_graph_snapshot(),
        }


class AnalyticalStudentModel(CognitiveStudentModel):
    """
    Careful, systematic student.

    Cognitive profile (research-grounded):
    - K₀: 0.62–0.75 (upper half of Koedinger range)
    - μ: 0.09–0.12 logits/opp (near population mean, low variance)
    - Attention span: 12–18 problems (~36–54 min, high focus)
    - Motivation: 0.65–0.82 (AMS: high intrinsic, low amotivation)
    - Low careless error rate (2–9%)
    - No overconfidence bias  -  conservative self-assessment
    - Prefers step-by-step problem solving (0.82–1.0)
    - Lower mind-wandering baseline (~0.20 vs population 0.30)
    - Asks for help when stuck but does not give up easily

    Both sub-agents (DecisionAgent + SolverAgent) use the "analytical"
    persona, making this student cautious in what they attempt and verbose
    in how they answer.
    """

    def __init__(
        self,
        student_id: str,
        concept_dim: int = 58,
        history_size: int = 50,
    ):
        from .llm_answer_generator import build_dual_agents_from_env
        decision_agent, solver_agent = build_dual_agents_from_env("analytical")

        super().__init__(
            student_id=student_id,
            initial_knowledge_level=np.random.uniform(0.62, 0.75),
            learning_rate=np.random.uniform(0.09, 0.12),
            attention_span=np.random.randint(12, 19),
            motivation_level=np.random.uniform(0.65, 0.82),
            student_type="analytical",
            concept_dim=concept_dim,
            decision_agent=decision_agent,
            solver_agent=solver_agent,
            history_size=history_size,
        )
        # Override learning-style parameters with analytical defaults
        self.prefers_step_by_step = np.random.uniform(0.82, 1.0)
        self.challenge_tolerance = np.random.uniform(0.25, 0.55)
        self.help_seeking_tendency = np.random.uniform(0.20, 0.45)
        self.careless_error_rate = np.random.uniform(0.02, 0.09)
        self.overconfidence_bias = 0.0
        # Analytical students are more focused (Smallwood & Schooler: activity
        # importance/interest reduces mind-wandering)
        self.mind_wandering_base = np.random.uniform(0.15, 0.25)


class IntuitiveStudentModel(CognitiveStudentModel):
    """
    Quick, pattern-matching student.

    Cognitive profile (research-grounded):
    - K₀: 0.55–0.72 (full Koedinger range, variable)
    - μ: 0.07–0.13 logits/opp (slightly wider variance)
    - Attention span: 5–8 problems (~15–24 min, low focus)
    - Motivation: 0.50–0.75 (AMS: moderate intrinsic, some amotivation)
    - High careless error rate (18–38%)
    - Overconfidence bias (0.10–0.28)  -  overestimates own ability
    - Dislikes showing work; jumps to answers (step-by-step 0.05–0.30)
    - Higher mind-wandering baseline (~0.35–0.45 vs population 0.30)
    - Rarely asks for help (relies on gut feel)
    - Will attempt problems even with partial concept knowledge

    Both sub-agents (DecisionAgent + SolverAgent) use the "intuitive"
    persona, making this student optimistic about attempting problems
    and terse/error-prone in their answers.
    """

    def __init__(
        self,
        student_id: str,
        concept_dim: int = 58,
        history_size: int = 50,
    ):
        from .llm_answer_generator import build_dual_agents_from_env
        decision_agent, solver_agent = build_dual_agents_from_env("intuitive")

        super().__init__(
            student_id=student_id,
            initial_knowledge_level=np.random.uniform(0.55, 0.72),
            learning_rate=np.random.uniform(0.07, 0.13),
            attention_span=np.random.randint(5, 9),
            motivation_level=np.random.uniform(0.50, 0.75),
            student_type="intuitive",
            concept_dim=concept_dim,
            decision_agent=decision_agent,
            solver_agent=solver_agent,
            history_size=history_size,
        )
        # Override learning-style parameters with intuitive defaults
        self.prefers_step_by_step = np.random.uniform(0.05, 0.30)
        self.challenge_tolerance = np.random.uniform(0.65, 0.92)
        self.help_seeking_tendency = np.random.uniform(0.10, 0.30)
        self.careless_error_rate = np.random.uniform(0.18, 0.38)
        self.overconfidence_bias = np.random.uniform(0.10, 0.28)
        # Intuitive students are more distractible (Smallwood & Schooler:
        # boredom and low engagement increase mind-wandering)
        self.mind_wandering_base = np.random.uniform(0.35, 0.45)


class StudentPopulation:
    """Manages a diverse population of student models"""

    def __init__(
        self,
        num_students: int = 1000,
        concept_dim: int = 50,
    ):
        self.num_students = num_students
        self.concept_dim = concept_dim
        self.students = []
        self._next_student_id = 0
        self._create_student_population()

    # Population-level type distribution (7 types)
    _TYPE_DISTRIBUTION = {
        'average': 0.30,
        'analytical': 0.15,
        'intuitive': 0.15,
        'fast_learner': 0.15,
        'struggling_learner': 0.15,
        'anxious_learner': 0.05,
        'confident_learner': 0.05,
    }

    def _create_student_population(self):
        """Create diverse student population."""
        for i in range(self.num_students):
            student_type = np.random.choice(
                list(self._TYPE_DISTRIBUTION.keys()),
                p=list(self._TYPE_DISTRIBUTION.values()),
            )
            student = self._create_student_by_type(f"student_{i:04d}", student_type)
            self.students.append(student)
            self._next_student_id = i + 1

    def _create_student_by_type(self, student_id: str, student_type: str) -> CognitiveStudentModel:
        """Create student with specific type characteristics.

        All student types use the dual-agent (DecisionAgent + SolverAgent)
        architecture backed by GPT-5 Nano.  Analytical and Intuitive types
        have dedicated subclasses with custom cognitive profiles.
        """
        from .llm_answer_generator import build_dual_agents_from_env

        # --- Dedicated subclasses ---
        if student_type == 'analytical':
            return AnalyticalStudentModel(
                student_id=student_id,
                concept_dim=self.concept_dim,
            )

        if student_type == 'intuitive':
            return IntuitiveStudentModel(
                student_id=student_id,
                concept_dim=self.concept_dim,
            )

        # --- All other types ---
        decision_agent, solver_agent = build_dual_agents_from_env(student_type)

        base_params: Dict[str, Any] = {
            'student_id': student_id,
            'student_type': student_type,
            'concept_dim': self.concept_dim,
            'decision_agent': decision_agent,
            'solver_agent': solver_agent,
        }

        # Research-grounded parameters (see module docstring for citations)
        # K₀: Koedinger mean ~0.65, range 0.55–0.75
        # μ: Koedinger ~0.10 logits/opp, low cross-student variance
        # Attention: Smallwood & Schooler, mapped to problem counts (~3 min/problem)
        # Motivation: AMS-derived composite φ, population mean ~0.68
        if student_type == 'fast_learner':
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.65, 0.78),
                'learning_rate': np.random.uniform(0.12, 0.16),
                'attention_span': np.random.randint(10, 17),
                'motivation_level': np.random.uniform(0.68, 0.85),
            }
        elif student_type == 'struggling_learner':
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.48, 0.60),
                'learning_rate': np.random.uniform(0.06, 0.09),
                'attention_span': np.random.randint(4, 8),
                'motivation_level': np.random.uniform(0.45, 0.62),
            }
        elif student_type == 'anxious_learner':
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.55, 0.68),
                'learning_rate': np.random.uniform(0.08, 0.12),
                'attention_span': np.random.randint(7, 12),
                'motivation_level': np.random.uniform(0.45, 0.65),
            }
        elif student_type == 'confident_learner':
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.60, 0.75),
                'learning_rate': np.random.uniform(0.08, 0.12),
                'attention_span': np.random.randint(7, 12),
                'motivation_level': np.random.uniform(0.65, 0.85),
            }
        else:  # average
            params = {
                **base_params,
                'initial_knowledge_level': np.random.uniform(0.58, 0.72),
                'learning_rate': np.random.uniform(0.08, 0.12),
                'attention_span': np.random.randint(8, 13),
                'motivation_level': np.random.uniform(0.58, 0.78),
            }

        student = CognitiveStudentModel(**params)

        # Override learning-style parameters with research-grounded type profiles.
        # Base class assigns random defaults (mind_wandering=0.30, help_seeking=0.1–0.7,
        # challenge_tolerance=0.2–0.8, careless_error=0.05–0.20)  -  identical for all
        # types, making type differentiation meaningless in the simulation.
        if student_type == 'fast_learner':
            student.mind_wandering_base   = np.random.uniform(0.15, 0.25)
            student.help_seeking_tendency = np.random.uniform(0.05, 0.20)
            student.challenge_tolerance   = np.random.uniform(0.70, 0.95)
            student.careless_error_rate   = np.random.uniform(0.05, 0.15)
        elif student_type == 'struggling_learner':
            student.mind_wandering_base   = np.random.uniform(0.40, 0.50)
            student.help_seeking_tendency = np.random.uniform(0.50, 0.80)
            student.challenge_tolerance   = np.random.uniform(0.10, 0.30)
            student.careless_error_rate   = np.random.uniform(0.15, 0.30)
        elif student_type == 'anxious_learner':
            student.mind_wandering_base   = np.random.uniform(0.35, 0.50)
            student.help_seeking_tendency = np.random.uniform(0.60, 0.90)
            student.challenge_tolerance   = np.random.uniform(0.10, 0.25)
            student.careless_error_rate   = np.random.uniform(0.10, 0.20)
        elif student_type == 'confident_learner':
            student.mind_wandering_base   = np.random.uniform(0.20, 0.30)
            student.help_seeking_tendency = np.random.uniform(0.10, 0.30)
            student.challenge_tolerance   = np.random.uniform(0.65, 0.90)
            student.careless_error_rate   = np.random.uniform(0.10, 0.25)
        else:  # average
            student.mind_wandering_base   = np.random.uniform(0.25, 0.35)
            student.help_seeking_tendency = np.random.uniform(0.25, 0.55)
            student.challenge_tolerance   = np.random.uniform(0.35, 0.65)
            student.careless_error_rate   = np.random.uniform(0.08, 0.18)

        return student

    def sample_student(self) -> CognitiveStudentModel:
        """Create a fresh student sampled from the population distribution."""
        student_type = np.random.choice(
            list(self._TYPE_DISTRIBUTION.keys()),
            p=list(self._TYPE_DISTRIBUTION.values()),
        )
        student_id = f"student_{self._next_student_id:04d}"
        self._next_student_id += 1
        return self._create_student_by_type(student_id, student_type)

    def get_random_student(self) -> CognitiveStudentModel:
        """Get a random student from the population"""
        return np.random.choice(self.students)

    def get_population_stats(self) -> Dict:
        """Get statistics about the student population"""
        type_counts = {}
        knowledge_levels = []
        motivation_levels = []
        learning_rates = []

        for student in self.students:
            # Count student types
            student_type = student.student_type
            type_counts[student_type] = type_counts.get(student_type, 0) + 1

            # Collect stats
            knowledge_levels.append(np.mean(student.knowledge_state))
            motivation_levels.append(student.motivation_level)
            learning_rates.append(student.learning_rate)

        return {
            'total_students': len(self.students),
            'type_distribution': type_counts,
            'knowledge_stats': {
                'mean': np.mean(knowledge_levels),
                'std': np.std(knowledge_levels),
                'min': np.min(knowledge_levels),
                'max': np.max(knowledge_levels)
            },
            'motivation_stats': {
                'mean': np.mean(motivation_levels),
                'std': np.std(motivation_levels),
                'min': np.min(motivation_levels),
                'max': np.max(motivation_levels)
            },
            'learning_rate_stats': {
                'mean': np.mean(learning_rates),
                'std': np.std(learning_rates),
                'min': np.min(learning_rates),
                'max': np.max(learning_rates)
            }
        }
