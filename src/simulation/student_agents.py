"""
Dual-agent architecture for student behavior simulation.

DecisionAgent   -  GPT-5 Nano agent that decides, using the student's full
                 knowledge graph, whether the student can solve a problem.
SolverAgent     -  GPT-5 Nano agent that generates the actual student answer,
                 conditioned on the Decision Agent's verdict and the student's
                 cognitive persona.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System-prompt templates (keyed by student_type)
# ---------------------------------------------------------------------------

_DECISION_SYSTEM_PROMPTS: Dict[str, str] = {
    "analytical": (
        "You are simulating an analytical math student who is cautious and systematic. "
        "This student claims they can solve a problem when they have reasonable mastery "
        "(mastery_score > 0.6) of MOST key concepts required (roughly 70% or more of them). "
        "They take a methodical approach and prefer reviewing prerequisites when concepts "
        "are clearly weak, but they DO attempt problems when most relevant concepts are "
        "adequately mastered, even if one or two are slightly weaker. "
        "Lean toward can_solve=true when the majority of key concepts are at mastery > 0.6. "
        "Lean toward can_solve=false only when MULTIPLE key concepts are clearly weak "
        "(mastery < 0.5) or completely unknown."
    ),
    "intuitive": (
        "You are simulating an intuitive math student who is quick and pattern-focused. "
        "This student is often overconfident: they will attempt most problems even when "
        "they only partially understand the required concepts, relying on pattern recognition. "
        "They tend to underestimate how often they make careless errors. Lean toward "
        "can_solve=true unless the student has almost no relevant knowledge."
    ),
    "fast_learner": (
        "You are simulating a fast-learning math student with strong knowledge retention. "
        "They can solve most medium-difficulty problems and adapt quickly. They will "
        "attempt challenging problems if at least some concepts are mastered."
    ),
    "struggling_learner": (
        "You are simulating a student who struggles with math but is still trying. "
        "They have lower confidence and frequently make conceptual or calculation mistakes, "
        "but they DO attempt most problems even when uncertain  -  that is how struggling "
        "learners actually behave in real classrooms. They will write something down and "
        "guess, even when they know they may be wrong.\n\n"
        "IMPORTANT  -  calibration of the inputs you will see:\n"
        "- Concept mastery_score values for this student typically sit in the 0.50–0.70 band. "
        "Treat 0.55+ as 'has some familiarity' (enough to attempt), 0.50–0.55 as 'shaky', "
        "and below 0.50 as 'clearly weak'. Do NOT expect masteries near 0.0; that almost "
        "never happens.\n"
        "- heuristic_prior.success_probability is already calibrated to this student's "
        "ability. Use it as your primary signal: probabilities of 0.20+ mean a real "
        "struggling student would still try (and likely get part of it wrong), "
        "not refuse.\n\n"
        "Decision rules:\n"
        "- Set can_solve=true whenever heuristic_prior.success_probability >= 0.20 "
        "AND motivation_level >= 0.35 AND consecutive_failures < 3. This is the "
        "DEFAULT case for struggling students and should fire on the majority of problems.\n"
        "- Set can_solve=false ONLY when the student would realistically refuse to attempt: "
        "heuristic_prior.success_probability < 0.20, OR motivation_level < 0.35, "
        "OR consecutive_failures >= 3 with no recent successes, OR the problem clearly "
        "requires a concept whose mastery is below 0.45 with all related concepts also weak.\n"
        "- predicted_outcome should usually be 'wrong' or 'partial' (NOT 'give_up') when "
        "can_solve=true  -  struggling students attempt and get answers wrong, they don't "
        "always give up. Reserve 'give_up' for the can_solve=false branch."
    ),
    "anxious_learner": (
        "You are simulating an anxious math student. Even when they know the material, "
        "anxiety lowers their effective confidence. They often second-guess themselves "
        "and may request help even on problems they could solve."
    ),
    "confident_learner": (
        "You are simulating a confident math student. They approach problems optimistically "
        "and rarely give up. They may slightly overestimate their ability but generally "
        "have good knowledge foundations."
    ),
    "average": (
        "You are simulating an average math student. They attempt problems with moderate "
        "confidence and sometimes ask for help when concepts are unfamiliar."
    ),
}

_SOLVER_SYSTEM_PROMPTS: Dict[str, str] = {
    "analytical": (
        "You write answers as a careful, analytical math student. "
        "Always show your work step by step using phrases like 'First I will...', "
        "'Next...', 'Therefore the answer is...'. "
        "If you cannot solve the problem, clearly state which specific concept or step "
        "you are stuck on and what prerequisite you need to review. "
        "Never guess  -  if you are uncertain, say so explicitly."
    ),
    "intuitive": (
        "You write answers as a quick, intuitive math student. "
        "Jump straight to the approach without lengthy preamble. "
        "Sometimes skip intermediate steps. "
        "Errors are usually small arithmetic mistakes (off by one, sign flip, wrong "
        "multiplication) rather than conceptual misunderstandings. "
        "Keep answers brief and direct."
    ),
    "fast_learner": (
        "You write answers as a capable, efficient student. "
        "Show key reasoning steps but skip trivial arithmetic. "
        "Confident tone, accurate most of the time."
    ),
    "struggling_learner": (
        "You write answers as a student who is unsure and often confused, but who still "
        "tries to put down an answer. "
        "Use phrases like 'I think...', 'I'm not sure but...', 'Maybe...'. "
        "Make conceptual errors  -  apply wrong formulas, mix up which numbers go where, "
        "drop a step, or confuse related ideas (e.g. add when you should multiply). "
        "When can_solve=true: you MUST commit to a concrete final_answer (a single number "
        "or short string), even if you suspect it is wrong. Do NOT leave final_answer null "
        "when can_solve=true  -  guess if you have to. Real struggling students write down "
        "an answer and move on. "
        "When can_solve=false: express frustration, explain which step lost you, and ask "
        "for help (final_answer must be null in that case)."
    ),
    "anxious_learner": (
        "You write answers as an anxious student. "
        "Even when you know the material you second-guess yourself. "
        "Write things like 'I hope this is right...' or 'I keep second-guessing step 2'. "
        "You may ask for confirmation mid-answer."
    ),
    "confident_learner": (
        "You write answers as a confident student. "
        "Tackle problems decisively. Show enough work to be clear but without "
        "over-explaining. Rarely express doubt."
    ),
    "average": (
        "You write answers as a typical math student. "
        "Moderate level of detail, standard approach. "
        "Sometimes makes small errors but generally follows correct procedure."
    ),
}

_DECISION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "can_solve": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "predicted_outcome": {
            "type": "string",
            "enum": ["success", "partial", "wrong", "give_up"],
        },
        "error_types": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "practice_targets": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
        "reasoning": {"type": "string", "maxLength": 280},
    },
    "required": [
        "can_solve",
        "confidence",
        "predicted_outcome",
        "error_types",
        "practice_targets",
        "reasoning",
    ],
    "additionalProperties": False,
}

_SOLVER_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_text": {"type": "string", "maxLength": 800},
        "final_answer": {"type": ["string", "null"], "maxLength": 120},
        "gave_up": {"type": "boolean"},
        "help_requested": {"type": "boolean"},
        "needs_more_training": {"type": "boolean"},
        "hint_text": {"type": "string", "maxLength": 320},
        "practice_targets": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
    },
    "required": [
        "answer_text",
        "final_answer",
        "gave_up",
        "help_requested",
        "needs_more_training",
        "hint_text",
        "practice_targets",
    ],
    "additionalProperties": False,
}


def _response_meta(response: Any) -> Dict[str, Any]:
    """Extract a small, log-friendly metadata summary from a Responses object."""
    response_id = getattr(response, "id", None)
    status = getattr(response, "status", None)
    model = getattr(response, "model", None)

    details = getattr(response, "incomplete_details", None)
    reason = None
    if isinstance(details, dict):
        reason = details.get("reason")
    elif details is not None:
        reason = getattr(details, "reason", None)

    return {
        "id": response_id,
        "status": status,
        "model": model,
        "incomplete_reason": reason,
    }


# ---------------------------------------------------------------------------
# DecisionAgent
# ---------------------------------------------------------------------------

class DecisionAgent:
    """
    GPT-5 Nano agent that decides whether a student can solve a given problem.

    Uses the student's complete knowledge graph state (concept mastery scores,
    recent history, attention, motivation) to make a realistic can_solve decision
    that reflects the student's cognitive profile.
    """

    def __init__(
        self,
        api_key: str,
        student_type: str = "average",
        config: Optional[Any] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for DecisionAgent. "
                "Install it: `pip install openai`."
            ) from exc

        from .llm_answer_generator import GPT5NanoConfig
        self.config = config or GPT5NanoConfig()
        self.client = OpenAI(api_key=api_key, timeout=self.config.timeout)
        self.student_type = student_type
        self.system_prompt = _DECISION_SYSTEM_PROMPTS.get(
            student_type, _DECISION_SYSTEM_PROMPTS["average"]
        )

    def __call__(
        self,
        problem: Dict[str, Any],
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Decide if the student can solve the problem.

        Returns a dict with:
          can_solve, confidence, predicted_outcome, error_types,
          practice_targets, reasoning
        """
        prompt = self._build_prompt(problem, behavior, prototype, context)
        max_retries = max(1, int(getattr(self.config, "max_retries", 3)))
        base_tokens = max(128, int(getattr(self.config, "max_output_tokens", 700)))
        prompt_chars = len(prompt)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            token_budget = min(4000, base_tokens * (2 ** attempt))
            logger.info(
                "[DecisionAgent] → OpenAI  student=%s  model=%s  attempt=%d/%d"
                "  token_budget=%d  prompt_chars=%d",
                self.student_type,
                self.config.model,
                attempt + 1,
                max_retries,
                token_budget,
                prompt_chars,
            )
            try:
                t0 = time.time()
                response = self.client.responses.create(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    reasoning={"effort": self.config.reasoning_effort},
                    text={
                        "verbosity": self.config.verbosity,
                        "format": {
                            "type": "json_schema",
                            "name": "decision_output",
                            "strict": True,
                            "schema": _DECISION_OUTPUT_SCHEMA,
                        },
                    },
                    max_output_tokens=token_budget,
                )
                meta = _response_meta(response)
                _raise_for_incomplete_response(response)
                raw = _extract_response_text(response)
                parsed = _parse_json_block(raw)
                result = self._normalize_decision(parsed, behavior, prototype)
                logger.info(
                    "[DecisionAgent] ← OK  %.2fs  response_id=%s  status=%s"
                    "  can_solve=%s  confidence=%.3f  outcome=%s  errors=%s",
                    time.time() - t0,
                    meta["id"],
                    meta["status"],
                    result["can_solve"],
                    result["confidence"],
                    result["predicted_outcome"],
                    result["error_types"],
                )
                logger.debug(
                    "[DecisionAgent] FULL RESPONSE JSON:\n%s",
                    json.dumps(result, indent=2, default=str),
                )
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[DecisionAgent] ← FAIL attempt %d/%d  token_budget=%d"
                    "  prompt_chars=%d  student=%s  error=%s",
                    attempt + 1,
                    max_retries,
                    token_budget,
                    prompt_chars,
                    self.student_type,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(0.3 * (attempt + 1))
        logger.error(
            "[DecisionAgent] ← FALLBACK after %d retries  student=%s"
            "  prompt_chars=%d  error=%s",
            max_retries,
            self.student_type,
            prompt_chars,
            last_exc,
        )
        return self._fallback_decision(behavior, prototype)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        problem: Dict[str, Any],
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        ess = context.get("epistemic_state_spec", {})
        student_state = context.get("student_state", {})
        heuristic = context.get("heuristic_prediction", {})
        persona = context.get("student_persona", {})

        # Summarise knowledge graph  -  keep only top-15 nodes by mastery_score
        kg_summary_raw: Dict[str, Any] = context.get("knowledge_graph_summary", {})
        kg_top = sorted(
            [
                {
                    "node": nid,
                    "type": v.get("type"),
                    "global_state": v.get("global_state"),
                    "mastery_score": round(float(v.get("mastery_score", 0.5)), 3),
                    "good": v.get("good_count", 0),
                    "bad": v.get("bad_count", 0),
                }
                for nid, v in kg_summary_raw.items()
            ],
            key=lambda x: x["mastery_score"],
            reverse=True,
        )[:15]

        # Recent history  -  last 10 attempts, now includes question text and student's answer
        recent: List[Dict] = context.get("recent_history", [])[-10:]
        recent_compact = [
            {
                "problem_id": r.get("problem_id"),
                "question_text": r.get("question_text", ""),
                "student_final_answer": r.get("student_final_answer"),
                "student_answer_text": r.get("student_answer_text", ""),
                "correct_answer": r.get("correct_answer"),
                "answer_correct": r.get("answer_correct"),
                "success": r.get("success"),
                "gave_up": r.get("gave_up", False),
                "help_requested": r.get("help_requested", False),
                "difficulty": round(float(r.get("difficulty_score", 0.5)), 2),
                "predicted_outcome": r.get("predicted_outcome"),
                "error_types": list(r.get("error_types", [])),
            }
            for r in recent
        ]

        # Past questions from the knowledge graph (question nodes with text)
        past_questions = context.get("past_questions_in_graph", [])[:10]

        payload = {
            "question": str(problem.get("question", "")).strip(),
            "difficulty": problem.get("difficulty_score"),
            "concept_names": prototype.get("concept_names", []),
            "sub_concept_names": prototype.get("sub_concept_names", []),
            "mapped_top_nodes": prototype.get("mapped_top_nodes", []),
            "graph_states_for_top_nodes": prototype.get("global_cognitive_states", {}),
            "mastery_ratio": round(float(prototype.get("mastery_ratio", 0.5)), 3),
            "knowledge_graph_top15": kg_top,
            "past_questions_seen": past_questions,
            "recent_attempts": recent_compact,
            "student_state": {
                "motivation_level": round(float(student_state.get("motivation_level", 0.5)), 3),
                "fatigue_level": round(float(student_state.get("fatigue_level", 0.0)), 3),
                "consecutive_failures": student_state.get("consecutive_failures", 0),
                "consecutive_successes": student_state.get("consecutive_successes", 0),
                "confidence_level": round(
                    float(student_state.get("confidence_level", 0.5)), 3
                ),
                "recent_success_rate": round(
                    float(student_state.get("recent_success_rate", 0.5)), 3
                ),
            },
            "student_persona": {
                "student_type": persona.get("student_type"),
                "prefers_step_by_step": round(
                    float(persona.get("prefers_step_by_step", 0.5)), 3
                ),
                "help_seeking_tendency": round(
                    float(persona.get("help_seeking_tendency", 0.3)), 3
                ),
                "careless_error_rate": round(
                    float(persona.get("careless_error_rate", 0.1)), 3
                ),
                "overconfidence_bias": round(
                    float(persona.get("overconfidence_bias", 0.0)), 3
                ),
                "challenge_tolerance": round(
                    float(persona.get("challenge_tolerance", 0.5)), 3
                ),
                "attention_span": int(persona.get("attention_span", 10)),
            },
            "heuristic_prior": {
                "predicted_outcome": heuristic.get("predicted_outcome"),
                "success_probability": round(
                    float(heuristic.get("success_probability", 0.5)), 3
                ),
                "error_types": heuristic.get("error_types", []),
            },
        }

        schema = (
            "{\n"
            '  "can_solve": true,\n'
            '  "confidence": 0.75,\n'
            '  "predicted_outcome": "success|partial|wrong|give_up",\n'
            '  "error_types": ["missing_concept", "careless_error"],\n'
            '  "practice_targets": ["concept::algebra", "sub_concept::linear_eq"],\n'
            '  "reasoning": "one sentence"\n'
            "}"
        )

        return (
            "You must decide whether this math student can solve the given problem "
            "based on their knowledge graph, past question history, and current cognitive state.\n\n"
            "Rules:\n"
            "- Examine mastery_score per concept node (0=unknown, 1=mastered).\n"
            "- Review recent_attempts: question text, what the student actually wrote "
            "(student_answer_text), their final answer, and whether it was correct.\n"
            "- Use past_questions_seen to understand which problem types the student has faced.\n"
            "- Repeated failures with the same error_types signals a persistent gap.\n"
            "- If can_solve=false, set practice_targets to the weakest concept nodes.\n"
            "- confidence must reflect actual mastery evidence, not just difficulty.\n"
            "- predicted_outcome must be one of: success, partial, wrong, give_up.\n"
            "- Return ONLY valid JSON with no markdown.\n\n"
            f"SCHEMA:\n{schema}\n\n"
            f"STUDENT_CONTEXT_JSON:\n{json.dumps(payload, ensure_ascii=True)}"
        )

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalize_decision(
        self,
        parsed: Dict[str, Any],
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
    ) -> Dict[str, Any]:
        mapped_nodes = list(prototype.get("mapped_top_nodes", []) or [])
        valid_outcomes = {"success", "partial", "wrong", "give_up"}

        can_solve = bool(parsed.get("can_solve", False))
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        predicted_outcome = str(parsed.get("predicted_outcome", "wrong"))
        if predicted_outcome not in valid_outcomes:
            predicted_outcome = behavior.get("predicted_outcome", "wrong")

        error_types_raw = parsed.get("error_types", [])
        if not isinstance(error_types_raw, list):
            error_types_raw = []
        error_types = [str(e).strip() for e in error_types_raw if str(e).strip()]
        if not error_types:
            error_types = list(behavior.get("error_types", ["unspecified_error"]))

        practice_targets_raw = parsed.get("practice_targets", [])
        if not isinstance(practice_targets_raw, list):
            practice_targets_raw = []
        practice_targets = [str(x).strip() for x in practice_targets_raw if str(x).strip()]
        if not practice_targets:
            practice_targets = mapped_nodes[:3]

        return {
            "can_solve": can_solve,
            "confidence": confidence,
            "predicted_outcome": predicted_outcome,
            "error_types": error_types,
            "practice_targets": practice_targets,
            "reasoning": str(parsed.get("reasoning", "")),
            "llm_failed": False,
        }

    def _fallback_decision(
        self,
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Emergency response when the LLM API is unreachable after all retries.

        Derives a best-effort decision from the statistical behavior prediction
        so the training episode can continue without crashing.  Sets
        ``llm_failed=True`` so callers can detect and log the incident.
        """
        predicted_outcome = behavior.get("predicted_outcome", "wrong")
        can_solve = predicted_outcome == "success"
        return {
            "can_solve": can_solve,
            "confidence": float(behavior.get("success_probability", 0.5)),
            "predicted_outcome": predicted_outcome,
            "error_types": list(behavior.get("error_types", ["unspecified_error"])),
            "practice_targets": list(prototype.get("mapped_top_nodes", [])[:3]),
            "reasoning": "LLM API unreachable  -  derived from statistical behavior prediction",
            "llm_failed": True,
        }


# ---------------------------------------------------------------------------
# SolverAgent
# ---------------------------------------------------------------------------

class SolverAgent:
    """
    GPT-5 Nano agent that generates the actual student answer.

    Conditioned on the Decision Agent's verdict (can_solve, error_types, etc.)
    and the student's persona, it produces an answer that is authentic to that
    student's cognitive style.
    """

    def __init__(
        self,
        api_key: str,
        student_type: str = "average",
        config: Optional[Any] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for SolverAgent. "
                "Install it: `pip install openai`."
            ) from exc

        from .llm_answer_generator import GPT5NanoConfig
        self.config = config or GPT5NanoConfig()
        self.client = OpenAI(api_key=api_key, timeout=self.config.timeout)
        self.student_type = student_type
        self.system_prompt = _SOLVER_SYSTEM_PROMPTS.get(
            student_type, _SOLVER_SYSTEM_PROMPTS["average"]
        )

    def __call__(
        self,
        problem: Dict[str, Any],
        decision: Dict[str, Any],
        student_persona: Dict[str, Any],
        previous_candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate the student's answer.

        Returns a dict with:
          answer_text, final_answer, gave_up, help_requested,
          needs_more_training, hint_text, practice_targets, can_solve
        """
        prompt = self._build_prompt(problem, decision, student_persona, previous_candidate)
        max_retries = max(1, int(getattr(self.config, "max_retries", 3)))
        base_tokens = max(128, int(getattr(self.config, "max_output_tokens", 700)))
        prompt_chars = len(prompt)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            token_budget = min(4000, base_tokens * (2 ** attempt))
            logger.info(
                "[SolverAgent]   → OpenAI  student=%s  can_solve=%s  attempt=%d/%d"
                "  token_budget=%d  prompt_chars=%d  has_prev=%s",
                self.student_type,
                decision.get("can_solve"),
                attempt + 1,
                max_retries,
                token_budget,
                prompt_chars,
                previous_candidate is not None,
            )
            try:
                t0 = time.time()
                response = self.client.responses.create(
                    model=self.config.model,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    reasoning={"effort": self.config.reasoning_effort},
                    text={
                        "verbosity": self.config.verbosity,
                        "format": {
                            "type": "json_schema",
                            "name": "solver_output",
                            "strict": True,
                            "schema": _SOLVER_OUTPUT_SCHEMA,
                        },
                    },
                    max_output_tokens=token_budget,
                )
                meta = _response_meta(response)
                _raise_for_incomplete_response(response)
                raw = _extract_response_text(response)
                parsed = _parse_json_block(raw)
                result = self._normalize_solution(parsed, decision, problem)
                logger.info(
                    "[SolverAgent]   ← OK  %.2fs  response_id=%s  status=%s"
                    "  gave_up=%s  help=%s  answer='%s'",
                    time.time() - t0,
                    meta["id"],
                    meta["status"],
                    result.get("gave_up"),
                    result.get("help_requested"),
                    str(result.get("final_answer", ""))[:60],
                )
                logger.debug(
                    "[SolverAgent] FULL RESPONSE JSON:\n%s",
                    json.dumps(result, indent=2, default=str),
                )
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[SolverAgent]   ← FAIL attempt %d/%d  token_budget=%d"
                    "  prompt_chars=%d  student=%s  can_solve=%s  has_prev=%s"
                    "  error=%s",
                    attempt + 1,
                    max_retries,
                    token_budget,
                    prompt_chars,
                    self.student_type,
                    decision.get("can_solve"),
                    previous_candidate is not None,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(0.3 * (attempt + 1))
        logger.error(
            "[SolverAgent]   ← FALLBACK after %d retries  student=%s"
            "  can_solve=%s  prompt_chars=%d  has_prev=%s  error=%s",
            max_retries,
            self.student_type,
            decision.get("can_solve"),
            prompt_chars,
            previous_candidate is not None,
            last_exc,
        )
        return self._fallback_solution(decision, problem)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        problem: Dict[str, Any],
        decision: Dict[str, Any],
        student_persona: Dict[str, Any],
        previous_candidate: Optional[Dict[str, Any]],
    ) -> str:
        can_solve = decision.get("can_solve", False)
        payload = {
            "question": str(problem.get("question", "")).strip(),
            "difficulty": problem.get("difficulty_score"),
            "concept_names": problem.get("concepts", []),
            "can_solve": can_solve,
            "confidence": decision.get("confidence"),
            "predicted_outcome": decision.get("predicted_outcome"),
            "error_types": decision.get("error_types", []),
            "practice_targets": decision.get("practice_targets", []),
            "student_persona": {
                "student_type": student_persona.get("student_type"),
                "prefers_step_by_step": student_persona.get("prefers_step_by_step"),
                "help_seeking_tendency": student_persona.get("help_seeking_tendency"),
                "careless_error_rate": student_persona.get("careless_error_rate", 0.1),
                "overconfidence_bias": student_persona.get("overconfidence_bias", 0.0),
            },
            "previous_candidate": previous_candidate,
        }

        schema = (
            "{\n"
            '  "answer_text": "student working or explanation",\n'
            '  "final_answer": "42 or null",\n'
            '  "gave_up": false,\n'
            '  "help_requested": false,\n'
            '  "needs_more_training": false,\n'
            '  "hint_text": "",\n'
            '  "practice_targets": []\n'
            "}"
        )

        rules = (
            "Rules:\n"
            "- If can_solve=true: provide answer_text and final_answer (must be non-null).\n"
            "- If can_solve=false: set final_answer=null, gave_up or help_requested=true, "
            "needs_more_training=true, and provide hint_text explaining what to review.\n"
            "- Match the student's cognitive style described in the system prompt.\n"
            "- If error_types includes 'careless_error', introduce a small numerical mistake.\n"
            "- If error_types includes 'missing_concept', show conceptual confusion.\n"
            "- Keep answer_text concise and student-like. No markdown.\n"
            "- practice_targets must come from the provided practice_targets list when possible.\n"
            "- Return ONLY valid JSON with no markdown fences.\n"
        )

        return (
            f"{rules}\n"
            f"SCHEMA:\n{schema}\n\n"
            f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=True)}"
        )

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalize_solution(
        self,
        parsed: Dict[str, Any],
        decision: Dict[str, Any],
        problem: Dict[str, Any],
    ) -> Dict[str, Any]:
        can_solve = decision.get("can_solve", False)
        practice_targets_fallback = list(decision.get("practice_targets", [])[:3])

        final_answer = parsed.get("final_answer")
        if isinstance(final_answer, str):
            final_answer = final_answer.strip() or None

        answer_text = str(parsed.get("answer_text", "")).strip()
        hint_text = str(parsed.get("hint_text", "")).strip()
        gave_up = bool(parsed.get("gave_up", False))
        help_requested = bool(parsed.get("help_requested", False))
        needs_more_training = bool(parsed.get("needs_more_training", False))

        practice_targets_raw = parsed.get("practice_targets", [])
        if not isinstance(practice_targets_raw, list):
            practice_targets_raw = []
        practice_targets = [str(x).strip() for x in practice_targets_raw if str(x).strip()]
        if not practice_targets:
            practice_targets = practice_targets_fallback

        # Enforce contract: if not can_solve, nullify answer and set flags
        if not can_solve:
            final_answer = None
            help_requested = True
            needs_more_training = True
            if not hint_text:
                hint_text = "I need to review the prerequisite concepts first."
        elif final_answer is None:
            # Claimed can_solve but gave no answer  -  downgrade
            can_solve = False
            help_requested = True
            needs_more_training = True
            if not hint_text:
                hint_text = "I tried but could not arrive at a final answer."

        return {
            "can_solve": can_solve,
            "answer_text": answer_text,
            "final_answer": final_answer,
            "gave_up": gave_up,
            "help_requested": help_requested,
            "needs_more_training": needs_more_training,
            "hint_text": hint_text,
            "practice_targets": practice_targets,
            "llm_failed": False,
        }

    def _fallback_solution(
        self,
        decision: Dict[str, Any],
        problem: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Emergency response when the LLM API is unreachable after all retries.

        Returns a null answer (final_answer=None) so grading marks it as
        incorrect.  Never leaks the correct answer.  Sets ``llm_failed=True``
        so callers can detect and log the incident.
        """
        practice_targets = list(decision.get("practice_targets", [])[:3])

        # When the LLM fails, the student cannot produce a real answer.
        # Return final_answer=None so grading marks it as incorrect.
        return {
            "can_solve": False,
            "answer_text": "I'm not sure how to approach this.",
            "final_answer": None,
            "gave_up": False,
            "help_requested": True,
            "needs_more_training": True,
            "hint_text": "Let me review the related concepts first.",
            "practice_targets": practice_targets,
            "llm_failed": True,
        }


# ---------------------------------------------------------------------------
# OllamaDecisionAgent / OllamaSolverAgent  (local model via chat completions)
# ---------------------------------------------------------------------------

class OllamaDecisionAgent(DecisionAgent):
    """
    DecisionAgent backed by a local Ollama model via the OpenAI-compatible
    chat/completions endpoint instead of the OpenAI Responses API.

    Inherits prompt construction and response normalisation from DecisionAgent.
    Only the API call path is different.
    """

    def __init__(
        self,
        student_type: str = "average",
        config: Optional[Any] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OllamaDecisionAgent. "
                "Install it: `pip install openai`."
            ) from exc

        from .llm_answer_generator import OllamaConfig
        self.config = config or OllamaConfig()
        self.client = OpenAI(
            api_key="ollama",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self.student_type = student_type
        self.system_prompt = _DECISION_SYSTEM_PROMPTS.get(
            student_type, _DECISION_SYSTEM_PROMPTS["average"]
        )

    def __call__(
        self,
        problem: Dict[str, Any],
        behavior: Dict[str, Any],
        prototype: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(problem, behavior, prototype, context)
        max_retries = max(1, int(getattr(self.config, "max_retries", 3)))
        base_tokens = max(128, int(getattr(self.config, "max_output_tokens", 1600)))
        prompt_chars = len(prompt)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            token_budget = min(8000, base_tokens * (2 ** attempt))
            logger.info(
                "[OllamaDecisionAgent] → Ollama  student=%s  model=%s  attempt=%d/%d"
                "  token_budget=%d  prompt_chars=%d",
                self.student_type,
                self.config.model,
                attempt + 1,
                max_retries,
                token_budget,
                prompt_chars,
            )
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        # /no_think disables qwen3 chain-of-thought so content is non-empty
                        {"role": "user", "content": f"/no_think {prompt}"},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=token_budget,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = _parse_json_block(raw)
                result = self._normalize_decision(parsed, behavior, prototype)
                logger.info(
                    "[OllamaDecisionAgent] ← OK  %.2fs  can_solve=%s  confidence=%.3f"
                    "  outcome=%s  errors=%s",
                    time.time() - t0,
                    result["can_solve"],
                    result["confidence"],
                    result["predicted_outcome"],
                    result["error_types"],
                )
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[OllamaDecisionAgent] ← FAIL attempt %d/%d  student=%s  error=%s",
                    attempt + 1,
                    max_retries,
                    self.student_type,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(0.3 * (attempt + 1))
        logger.error(
            "[OllamaDecisionAgent] ← FALLBACK after %d retries  student=%s  error=%s",
            max_retries,
            self.student_type,
            last_exc,
        )
        return self._fallback_decision(behavior, prototype)


class OllamaSolverAgent(SolverAgent):
    """
    SolverAgent backed by a local Ollama model via the OpenAI-compatible
    chat/completions endpoint instead of the OpenAI Responses API.

    Inherits prompt construction and response normalisation from SolverAgent.
    Only the API call path is different.
    """

    def __init__(
        self,
        student_type: str = "average",
        config: Optional[Any] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OllamaSolverAgent. "
                "Install it: `pip install openai`."
            ) from exc

        from .llm_answer_generator import OllamaConfig
        self.config = config or OllamaConfig()
        self.client = OpenAI(
            api_key="ollama",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        self.student_type = student_type
        self.system_prompt = _SOLVER_SYSTEM_PROMPTS.get(
            student_type, _SOLVER_SYSTEM_PROMPTS["average"]
        )

    def __call__(
        self,
        problem: Dict[str, Any],
        decision: Dict[str, Any],
        student_persona: Dict[str, Any],
        previous_candidate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(problem, decision, student_persona, previous_candidate)
        max_retries = max(1, int(getattr(self.config, "max_retries", 3)))
        base_tokens = max(128, int(getattr(self.config, "max_output_tokens", 1600)))
        prompt_chars = len(prompt)
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            token_budget = min(8000, base_tokens * (2 ** attempt))
            logger.info(
                "[OllamaSolverAgent]   → Ollama  student=%s  can_solve=%s  attempt=%d/%d"
                "  token_budget=%d  prompt_chars=%d  has_prev=%s",
                self.student_type,
                decision.get("can_solve"),
                attempt + 1,
                max_retries,
                token_budget,
                prompt_chars,
                previous_candidate is not None,
            )
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        # /no_think disables qwen3 chain-of-thought so content is non-empty
                        {"role": "user", "content": f"/no_think {prompt}"},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=token_budget,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = _parse_json_block(raw)
                result = self._normalize_solution(parsed, decision, problem)
                logger.info(
                    "[OllamaSolverAgent]   ← OK  %.2fs  gave_up=%s  help=%s  answer='%s'",
                    time.time() - t0,
                    result.get("gave_up"),
                    result.get("help_requested"),
                    str(result.get("final_answer", ""))[:60],
                )
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[OllamaSolverAgent]   ← FAIL attempt %d/%d  student=%s  error=%s",
                    attempt + 1,
                    max_retries,
                    self.student_type,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(0.3 * (attempt + 1))
        logger.error(
            "[OllamaSolverAgent]   ← FALLBACK after %d retries  student=%s  error=%s",
            max_retries,
            self.student_type,
            last_exc,
        )
        return self._fallback_solution(decision, problem)


# ---------------------------------------------------------------------------
# Shared JSON parsing helper
# ---------------------------------------------------------------------------

def _raise_for_incomplete_response(response: Any) -> None:
    """Raise when the API returns an incomplete response payload."""
    status = getattr(response, "status", None)
    if status is None and isinstance(response, dict):
        status = response.get("status")
    if status != "incomplete":
        return

    details = getattr(response, "incomplete_details", None)
    if details is None and isinstance(response, dict):
        details = response.get("incomplete_details")

    reason = None
    if isinstance(details, dict):
        reason = details.get("reason")
    else:
        reason = getattr(details, "reason", None)

    if reason:
        response_id = getattr(response, "id", None)
        if response_id:
            raise ValueError(f"Incomplete LLM response ({reason}) [response_id={response_id}]")
        raise ValueError(f"Incomplete LLM response ({reason})")
    raise ValueError("Incomplete LLM response")


def _extract_response_text(response: Any) -> str:
    """Get textual content from a Responses API payload."""
    raw = str(getattr(response, "output_text", "") or "").strip()
    if raw:
        return raw

    output_items = getattr(response, "output", None)
    if not output_items and isinstance(response, dict):
        output_items = response.get("output", [])

    chunks: List[str] = []
    for item in output_items or []:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "message":
            continue
        content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
        for part in content or []:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type not in {"output_text", "text"}:
                continue
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if text:
                chunks.append(str(text))

    # Concatenate directly; inserting separators can corrupt JSON when the API
    # streams one JSON object in multiple text chunks.
    return "".join(chunks).strip()


def _parse_json_block(text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from LLM output."""
    content = (text or "").strip()
    if not content:
        raise ValueError("Empty LLM response")

    # Strip markdown fences
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        content = content[start:end].strip()

    # Find JSON boundaries
    if not content.startswith("{"):
        idx = content.find("{")
        if idx >= 0:
            content = content[idx:]
    if not content.endswith("}"):
        idx = content.rfind("}")
        if idx >= 0:
            content = content[: idx + 1]

    return json.loads(content)
