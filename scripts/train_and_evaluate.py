#!/usr/bin/env python3
"""
Train a PPO tutoring agent and run comprehensive research evaluation.

Pipeline:
  1. Load problems from API (shared between train and test)
  2. Pre-training snapshot  -  capture initial student knowledge distribution
  3. Train PPO on training population (seed=42 by default)
  4. Evaluate on held-out test population (seed=999 by default)
       • Full NLG / concept-domain / quartile analysis
       • Compare against baselines
  5. Save JSON + text summary to outputs/train_and_eval_<timestamp>/

Usage:
    venv310/bin/python scripts/train_and_evaluate.py \\
        --api-base-url http://localhost:8000 \\
        --timesteps 100000 \\
        --eval-episodes 100

By default this loads the full problem bank. Add --balanced only for
diagnostic difficulty-stratified runs.

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Force line-buffered stdout so progress prints appear immediately even when
# output is piped to a file (Python buffers stdout by default in non-tty mode).
sys.stdout.reconfigure(line_buffering=True)

# ── repo root on sys.path so src/ modules resolve ───────────────────────────
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from rl_agent.tutoring_env import SimulatedTutoringEnv
from rl_agent.evaluation_framework import (
    EvaluationConfig,
    evaluate_policy,
    _random_policy,
    _zpd_oracle_policy,
    _run_single_policy,
    _summarise_logs,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-training snapshot
# ─────────────────────────────────────────────────────────────────────────────

def _pre_training_snapshot(
    env: SimulatedTutoringEnv,
    n_episodes: int = 30,
) -> Dict:
    """Collect initial student state distributions BEFORE any tutoring.

    Uses the random policy for exactly one step per student so we can
    read the knowledge state immediately after reset  -  giving us the
    "student arrives at school" baseline distribution.
    """
    initial_knowledge: List[float] = []
    initial_motivation: List[float] = []
    types_seen: Dict[str, int] = {}

    random_fn = _random_policy(env)
    cfg = EvaluationConfig(
        episodes=n_episodes,
        max_steps=1,           # one step only  -  just to enter the episode loop
        compare_baselines=False,
        capture_concept_vectors=True,
    )
    logs = _run_single_policy(env, random_fn, cfg)

    for lg in logs:
        initial_knowledge.append(lg.initial_knowledge)
        initial_motivation.append(lg.initial_motivation)
        types_seen[lg.student_type] = types_seen.get(lg.student_type, 0) + 1

    return {
        "n_students": len(logs),
        "mean_initial_knowledge": float(np.mean(initial_knowledge)),
        "std_initial_knowledge":  float(np.std(initial_knowledge)),
        "min_initial_knowledge":  float(np.min(initial_knowledge)),
        "max_initial_knowledge":  float(np.max(initial_knowledge)),
        "mean_initial_motivation": float(np.mean(initial_motivation)),
        "initial_knowledge_by_type": {
            t: float(np.mean([lg.initial_knowledge for lg in logs
                               if lg.student_type == t]))
            for t in sorted(types_seen.keys())
        },
        "student_type_counts": types_seen,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_pre_snapshot(snap: Dict) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  PRE-TRAINING SNAPSHOT  (student population  -  no tutoring yet)")
    print(sep)
    print(f"  Students sampled      : {snap['n_students']}")
    print(f"  Mean initial knowledge: {snap['mean_initial_knowledge']:.3f} "
          f"± {snap['std_initial_knowledge']:.3f}")
    print(f"  Knowledge range       : [{snap['min_initial_knowledge']:.3f}, "
          f"{snap['max_initial_knowledge']:.3f}]")
    print(f"  Mean initial motivation: {snap['mean_initial_motivation']:.3f}")
    print(f"\n  {'Student type':<24} {'Init knowledge':>14}")
    print("  " + "-" * 40)
    for stype, kval in snap["initial_knowledge_by_type"].items():
        count = snap["student_type_counts"].get(stype, 0)
        print(f"  {stype:<24} {kval:>14.3f}  (n={count})")
    print()


def _print_training_config(args: argparse.Namespace, env: SimulatedTutoringEnv) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  PPO TRAINING   -  Personalized Tutoring Agent")
    print(sep)
    print(f"  API URL          : {args.api_base_url}")
    print(f"  Problems loaded  : {len(env.problem_bank.problems)}")
    print(f"  Concept dim      : {env.concept_dim}")
    print(f"  Action dim       : {env.action_dim}")
    print(f"  State dim        : {env.state_dim}")
    print(f"  Max steps/ep     : {env.max_steps}")
    print(f"  Training seed    : {args.train_seed}")
    print(f"  Test seed        : {args.test_seed}")
    print(f"  Training steps   : {args.timesteps:,}")
    print(f"  Parallel envs    : {args.n_envs}")
    print(f"  Difficulty-balanced bank: {args.balanced}")

    diffs = [p.difficulty_score for p in env.problem_bank.problems]
    n_easy = sum(1 for d in diffs if d <= 0.55)
    n_med  = sum(1 for d in diffs if 0.55 < d <= 0.72)
    n_hard = sum(1 for d in diffs if d > 0.72)
    print(f"  Difficulty dist  : easy={n_easy}, med={n_med}, hard={n_hard} "
          f"(mean={np.mean(diffs):.3f})")
    print()


def _print_training_done(timesteps: int, elapsed: float) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  TRAINING COMPLETE")
    print(sep)
    print(f"  Timesteps trained : {timesteps:,}")
    print(f"  Wall-clock time   : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput        : {timesteps / elapsed:.0f} steps/sec")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO and run comprehensive tutoring-agent evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── environment ────────────────────────────────────────────────────────
    parser.add_argument("--api-base-url", default="http://localhost:8000",
                        help="RL-Math API base URL")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="Limit number of problems loaded (None = all)")
    parser.add_argument("--max-steps", type=int, default=12,
                        help="Max problems per episode (hard ceiling; cognitive exhaustion "
                             "terminates earlier for low-attention-span student types)")
    parser.add_argument("--balanced", action="store_true",
                        help="Use optional difficulty-balanced diagnostic problem sampling")
    parser.add_argument("--num-students", type=int, default=200,
                        help="Students in each simulated population")

    # ── train/test split ───────────────────────────────────────────────────
    parser.add_argument("--train-seed", type=int, default=42,
                        help="RNG seed for training environment / population")
    parser.add_argument("--test-seed", type=int, default=999,
                        help="RNG seed for held-out test population")

    # ── training hyperparameters ───────────────────────────────────────────
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total PPO training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel training environments")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO rollout steps per env before update")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.02,
                        help="Entropy coefficient (exploration)")
    parser.add_argument("--net-arch", nargs="+", type=int, default=[256, 256],
                        help="Policy network hidden layer sizes")
    parser.add_argument("--tensorboard", default=None,
                        help="TensorBoard log directory (optional)")

    # ── evaluation ─────────────────────────────────────────────────────────
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Episodes for main policy evaluation on test set")
    parser.add_argument("--baseline-episodes", type=int, default=50,
                        help="Episodes per baseline policy (faster)")
    parser.add_argument("--mastery-threshold", type=float, default=0.70,
                        help="Knowledge fraction considered exam-ready")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline comparison (faster)")
    parser.add_argument("--pre-snapshot-episodes", type=int, default=50,
                        help="Episodes for pre-training snapshot (0 = skip)")

    # ── checkpoints ────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save model + run evaluation every N timesteps (0 = disable)")
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=14,
                        help="Episodes for full evaluate_policy at each checkpoint "
                             "(probe set of 14 fixed students always runs regardless; "
                             "full eval only runs when this is >= 14)")

    # ── output ─────────────────────────────────────────────────────────────
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: outputs/train_and_eval_<timestamp>)")
    parser.add_argument("--model-name", default="ppo_tutor",
                        help="Model filename base (no extension)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training; load model from --load-model for eval only")
    parser.add_argument("--load-model", default=None,
                        help="Path to .zip model to load (use with --skip-training)")
    parser.add_argument("--resume-from", default=None,
                        help="Path to .zip model to resume training from (continues learning)")
    parser.add_argument("--step-offset", type=int, default=0,
                        help="Effective steps already trained before this run "
                             "(used to number checkpoint folders correctly when resuming)")

    args = parser.parse_args()

    # ── output directory ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "outputs" / f"train_and_eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── detailed step-by-step log file ─────────────────────────────────────
    # Keep high-volume simulation logs out of stdout/stderr and route them
    # into a dedicated file for post-hoc debugging.
    step_log_path = out_dir / "training_steps.log"
    file_handler = logging.FileHandler(step_log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(name)-40s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler], force=True)
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "stable_baselines3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"Step-by-step log → {step_log_path}", flush=True)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CallbackList, BaseCallback
    except ImportError:
        print("ERROR: stable-baselines3 is required.\n"
              "Install with: pip install stable-baselines3")
        sys.exit(1)

    # ── checkpoint + mid-training evaluation callback ──────────────────────
    class CheckpointEvalCallback(BaseCallback):
        """
        Every `checkpoint_interval` timesteps:
          1. Saves model to  out_dir/checkpoints/step_XXXXXXX/model.zip
          2. Evaluates on a FIXED probe set (same students at every checkpoint)
             so NLG comparisons across checkpoints are meaningful, not noise.
          3. Saves evaluation_report.json + metrics_summary.txt to same dir

        Fixed probe set design:
          - 2 students per type × 7 types = 14 students, generated once at init.
          - Same 14 students reused identically at every checkpoint.
          - Eliminates the random-student sampling variance that made earlier
            checkpoint NLG values incomparable (e.g. step 2048 looked good
            because 2 out of 5 random episodes hit confident_learners).
        """

        def __init__(
            self,
            eval_env: SimulatedTutoringEnv,
            out_dir: Path,
            checkpoint_interval: int,
            eval_episodes: int,
            mastery_threshold: float,
            pre_snap: Optional[Dict],
            verbose: int = 1,
        ):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.out_dir = out_dir
            self.checkpoint_interval = checkpoint_interval
            self.eval_episodes = eval_episodes
            self.mastery_threshold = mastery_threshold
            self.pre_snap = pre_snap
            self._last_checkpoint_step = 0
            self._best_nlg_score: float = -float("inf")
            self._best_nlg_step: int = -1
            self._step_offset: int = 0

            # Build fixed probe set: 2 students per type = 14 students total.
            # Generated ONCE here and reused at every checkpoint so NLG values
            # across checkpoints are directly comparable (same students, same
            # starting K-states).  Without this, 5 random episodes per checkpoint
            # gave incomparable results  -  step 2048 looked good only because it
            # happened to draw 2 confident_learners out of 5.
            _probe_types = [
                "average", "analytical", "intuitive",
                "fast_learner", "struggling_learner",
                "anxious_learner", "confident_learner",
            ]
            self._probe_students = []
            for stype in _probe_types:
                for _ in range(2):  # 2 per type
                    self._probe_students.append(
                        eval_env.student_population._create_student_by_type(
                            f"probe_{stype}_{_}", stype
                        )
                    )
            print(
                f"Probe set: {len(self._probe_students)} fixed students "
                f"({len(_probe_types)} types × 2)  -  reused at every checkpoint",
                flush=True,
            )

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_interval:
                self._last_checkpoint_step = self.num_timesteps
                self._run_checkpoint(self.num_timesteps)
            return True  # always continue training

        def _run_probe_eval(self, step: int) -> Dict:
            """Run the fixed probe set and return per-type and aggregate metrics."""
            env = self.eval_env
            max_steps = env.max_steps
            concept_dim = env.concept_dim

            nlg_by_type: Dict[str, List[float]] = {}
            gain_by_type: Dict[str, List[float]] = {}
            giveup_by_type: Dict[str, List[float]] = {}
            all_nlg: List[float] = []
            all_gain: List[float] = []
            all_reward: List[float] = []
            all_giveup: List[float] = []

            for student in self._probe_students:
                stype = student.student_type
                result = env.reset(options={"probe_student": student})
                obs = result[0] if isinstance(result, tuple) else result
                initial_k = float(np.mean(obs[:concept_dim]))

                ep_reward = 0.0
                ep_giveup = False
                done = False
                s = 0
                while not done and s < max_steps:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result
                    ep_reward += float(reward)
                    if info.get("gave_up"):
                        ep_giveup = True
                    s += 1

                final_k = float(np.mean(obs[:concept_dim]))
                gain = final_k - initial_k
                denom = max(1.0 - initial_k, 1e-3)
                nlg = float(np.clip(gain / denom, -1.0, 1.0))

                nlg_by_type.setdefault(stype, []).append(nlg)
                gain_by_type.setdefault(stype, []).append(gain)
                giveup_by_type.setdefault(stype, []).append(float(ep_giveup))
                all_nlg.append(nlg)
                all_gain.append(gain)
                all_reward.append(ep_reward)
                all_giveup.append(float(ep_giveup))

            return {
                "mean_nlg": float(np.mean(all_nlg)),
                "mean_gain": float(np.mean(all_gain)),
                "mean_reward": float(np.mean(all_reward)),
                "give_up_rate": float(np.mean(all_giveup)),
                "nlg_by_type": {t: float(np.mean(v)) for t, v in nlg_by_type.items()},
                "gain_by_type": {t: float(np.mean(v)) for t, v in gain_by_type.items()},
                "all_types_positive_nlg": all(
                    float(np.mean(v)) > 0 for v in nlg_by_type.values()
                ),
            }

        def _run_checkpoint(self, step: int) -> None:
            step = step + self._step_offset
            ckpt_dir = self.out_dir / "checkpoints" / f"step_{step:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save model
            self.model.save(str(ckpt_dir / "model"))
            sep = "=" * 66
            print(f"\n{sep}", flush=True)
            print(f"  CHECKPOINT  step={step:,}  →  {ckpt_dir}", flush=True)
            print(sep, flush=True)

            # 2. Run fixed-probe evaluation (reproducible across checkpoints)
            try:
                t0 = time.time()
                probe_metrics = self._run_probe_eval(step)
                elapsed = time.time() - t0

                mean_nlg = probe_metrics["mean_nlg"]
                mean_gain = probe_metrics["mean_gain"]
                give_up_rate = probe_metrics["give_up_rate"]
                all_positive = probe_metrics["all_types_positive_nlg"]

                print(
                    f"  [PROBE]  NLG={mean_nlg:+.4f}  gain={mean_gain:+.4f}  "
                    f"give_up={give_up_rate:.1%}  all_types_positive={all_positive}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
                print(f"  Per type:", flush=True)
                for stype, nlg_val in sorted(probe_metrics["nlg_by_type"].items()):
                    gain_val = probe_metrics["gain_by_type"].get(stype, 0.0)
                    print(f"    {stype:<22} NLG={nlg_val:+.4f}  gain={gain_val:+.4f}", flush=True)

                # 3a. Save probe metrics JSON
                probe_report = {
                    "checkpoint_step": step,
                    "eval_type": "fixed_probe",
                    "n_probe_students": len(self._probe_students),
                    "eval_time_s": round(elapsed, 2),
                    **probe_metrics,
                }
                with open(ckpt_dir / "probe_metrics.json", "w") as fh:
                    json.dump(probe_report, fh, indent=2, default=str)

                # 3b. Text summary
                with open(ckpt_dir / "metrics_summary.txt", "w") as fh:
                    fh.write(f"Checkpoint step      : {step:,}\n")
                    fh.write(f"Eval type            : fixed_probe (14 students, same every ckpt)\n")
                    fh.write(f"Eval time            : {elapsed:.1f}s\n")
                    fh.write(f"\n--- Learning Effectiveness (Fixed Probe) ---\n")
                    fh.write(f"Mean NLG             : {mean_nlg:+.4f}\n")
                    fh.write(f"Mean knowledge gain  : {mean_gain:+.4f}\n")
                    fh.write(f"All types positive   : {all_positive}\n")
                    fh.write(f"Give-up rate         : {give_up_rate:.1%}\n")
                    fh.write(f"Mean reward          : {probe_metrics['mean_reward']:.2f}\n")
                    fh.write(f"\n--- NLG by Student Type ---\n")
                    for stype, nlg_val in sorted(probe_metrics["nlg_by_type"].items()):
                        fh.write(f"  {stype:<22} {nlg_val:+.4f}\n")

                print(f"  Saved probe report → {ckpt_dir}", flush=True)

                # Save pedagogically-best model by NLG
                if mean_nlg > self._best_nlg_score:
                    self._best_nlg_score = mean_nlg
                    self._best_nlg_step = step
                    best_nlg_dir = self.out_dir / "best_nlg_model"
                    best_nlg_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(best_nlg_dir / "model"))
                    with open(best_nlg_dir / "best_nlg_info.json", "w") as _fh:
                        json.dump({
                            "best_nlg_step": step,
                            "best_nlg_score": round(self._best_nlg_score, 6),
                            "mean_knowledge_gain": mean_gain,
                            "give_up_rate": give_up_rate,
                            "all_types_positive_nlg": all_positive,
                            "nlg_by_type": probe_metrics["nlg_by_type"],
                        }, _fh, indent=2)
                    print(
                        f"  ★ New best NLG model  NLG={self._best_nlg_score:+.4f}"
                        f"  step={step}  → best_nlg_model/",
                        flush=True,
                    )

                # Also run full evaluate_policy for richer checkpoint reports
                # (only if eval_episodes is set high enough to be meaningful)
                if self.eval_episodes >= 14:
                    ckpt_eval_config = EvaluationConfig(
                        episodes=self.eval_episodes,
                        max_steps=self.eval_env.max_steps,
                        mastery_threshold=self.mastery_threshold,
                        window_splits=3,
                        compare_baselines=False,
                        capture_concept_vectors=True,
                    )
                    t1 = time.time()
                    summary = evaluate_policy(self.eval_env, self.model, ckpt_eval_config)
                    elapsed2 = time.time() - t1
                    print(
                        f"  [FULL]   NLG={summary.mean_nlg:+.4f}  "
                        f"gain={summary.mean_knowledge_gain:+.4f}  "
                        f"ready={summary.exam_readiness.fraction_ready:.1%}  "
                        f"reward={summary.mean_reward:.2f}  ({elapsed2:.0f}s)",
                        flush=True,
                    )
                    report_dict = summary.to_dict()
                    report_dict["meta"] = {
                        "checkpoint_step": step,
                        "eval_episodes": self.eval_episodes,
                        "eval_time_s": round(elapsed2, 2),
                        "probe_metrics": probe_metrics,
                    }
                    if self.pre_snap:
                        report_dict["pre_training_snapshot"] = self.pre_snap
                    with open(ckpt_dir / "evaluation_report.json", "w") as fh:
                        json.dump(report_dict, fh, indent=2, default=str)
                    summary = summary  # used below for best-nlg check via full eval

            except Exception as exc:
                import traceback
                print(f"  WARNING: checkpoint eval failed at step {step}: {exc}", flush=True)
                traceback.print_exc()

            print(f"{sep}\n", flush=True)


    # ── build test environment (held-out population) ───────────────────────
    print(f"\nLoading problems from {args.api_base_url} ...")
    common_env_kwargs = dict(
        api_base_url=args.api_base_url,
        max_problems=args.max_problems,
        max_steps=args.max_steps,
        num_students=args.num_students,
        balanced=args.balanced,
    )

    test_env = SimulatedTutoringEnv(**common_env_kwargs, seed=args.test_seed)
    _print_training_config(args, test_env)

    # ── pre-training snapshot ──────────────────────────────────────────────
    pre_snap: Optional[Dict] = None
    if args.pre_snapshot_episodes > 0:
        print(f"Running pre-training snapshot ({args.pre_snapshot_episodes} episodes) ...")
        pre_snap = _pre_training_snapshot(test_env, n_episodes=args.pre_snapshot_episodes)
        _print_pre_snapshot(pre_snap)

    # ── train or load model ────────────────────────────────────────────────
    if args.skip_training:
        if not args.load_model:
            print("ERROR: --skip-training requires --load-model <path>")
            sys.exit(1)
        print(f"Loading model from {args.load_model} ...")
        model = PPO.load(args.load_model)
        training_time = 0.0
    else:
        # Build training environment (separate seed → different student population)
        print(f"Building training environment (seed={args.train_seed}) ...")

        def _make_train_env(seed_offset: int):
            def _init():
                return SimulatedTutoringEnv(
                    **common_env_kwargs, seed=args.train_seed + seed_offset
                )
            return _init

        vec_env = DummyVecEnv([_make_train_env(i) for i in range(args.n_envs)])

        if args.resume_from:
            print(f"Resuming training from {args.resume_from} ...")
            model = PPO.load(args.resume_from, env=vec_env)
            # Override learning rate and entropy coef in case hyperparams changed
            model.learning_rate = args.learning_rate
            model.ent_coef = args.ent_coef
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=0.99,
                ent_coef=args.ent_coef,
                clip_range=0.2,
                max_grad_norm=0.5,
                n_epochs=10,
                policy_kwargs=dict(net_arch=args.net_arch),
                verbose=1,
                seed=args.train_seed,
                tensorboard_log=args.tensorboard,
                device="cpu",  # MLP policy runs faster on CPU
            )

        # NOTE: SB3's built-in EvalCallback is intentionally NOT used here.
        # It saves best_model/ by mean reward with only 5 random episodes on the
        # training-seed env  -  which selected reward-hacking checkpoints (step 2048
        # looked great with reward=+69.7 from 2 lucky confident_learner episodes
        # out of 5).  The CheckpointEvalCallback below replaces it with a fixed
        # probe set (14 students, same every checkpoint) and NLG-based selection.
        callbacks = []
        if args.checkpoint_interval > 0:
            ckpt_callback = CheckpointEvalCallback(
                eval_env=test_env,
                out_dir=out_dir,
                checkpoint_interval=args.checkpoint_interval,
                eval_episodes=args.checkpoint_eval_episodes,
                mastery_threshold=args.mastery_threshold,
                pre_snap=pre_snap,
                verbose=1,
            )
            ckpt_callback._step_offset = args.step_offset
            callbacks.append(ckpt_callback)
            print(f"Checkpoint eval every {args.checkpoint_interval:,} steps  "
                  f"(fixed probe: 14 students; full eval if episodes≥14)", flush=True)

        print("\n" + "=" * 66)
        print("  STARTING PPO TRAINING")
        print("=" * 66 + "\n")
        t0 = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            log_interval=10,
        )
        training_time = time.time() - t0
        _print_training_done(args.timesteps, training_time)

        model_path = out_dir / args.model_name
        model.save(str(model_path))
        print(f"Saved trained model → {model_path}.zip")

    # ── comprehensive evaluation on held-out test population ──────────────
    print("\n" + "=" * 66)
    print(f"  EVALUATING on held-out test population (seed={args.test_seed})")
    print("=" * 66)
    print(f"  Episodes           : {args.eval_episodes}")
    print(f"  Mastery threshold  : {args.mastery_threshold:.0%}")
    print(f"  Baselines          : {'yes (5 policies)' if not args.no_baselines else 'skipped'}")
    print(f"  Baseline episodes  : {args.baseline_episodes}")
    print()

    eval_config = EvaluationConfig(
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        mastery_threshold=args.mastery_threshold,
        window_splits=3,
        compare_baselines=not args.no_baselines,
        baseline_episodes=args.baseline_episodes,
        capture_concept_vectors=True,
    )

    t_eval_start = time.time()
    summary = evaluate_policy(test_env, model, eval_config)
    eval_time = time.time() - t_eval_start

    # ── print research report ──────────────────────────────────────────────
    report_title = (
        f"PPO PERSONALIZED TUTOR  -  TEST SET EVALUATION  "
        f"(train_seed={args.train_seed}, test_seed={args.test_seed})"
    )
    summary.print_report(title=report_title)

    # ── pre → post comparison ──────────────────────────────────────────────
    if pre_snap:
        sep = "=" * 66
        print(f"\n{sep}")
        print("  PRE → POST TUTORING COMPARISON")
        print(sep)
        pre_mean = pre_snap["mean_initial_knowledge"]
        post_mean = summary.exam_readiness.mean_final_knowledge
        overall_gain = post_mean - pre_mean
        from_zero = pre_snap["mean_initial_knowledge"]
        nlg_pop = (overall_gain / max(1.0 - from_zero, 1e-3))
        print(f"  Population mean knowledge BEFORE tutoring : {pre_mean:.3f}")
        print(f"  Population mean knowledge AFTER  tutoring : {post_mean:.3f}")
        print(f"  Overall gain                              : {overall_gain:+.4f}")
        print(f"  Population NLG                            : {nlg_pop:+.4f}")
        print(f"\n  {'Student type':<24} {'Before':>8}  {'After':>8}  {'Gain':>7}")
        print("  " + "-" * 52)
        for ts in summary.personalization.type_summaries:
            before = pre_snap["initial_knowledge_by_type"].get(ts.student_type, float("nan"))
            print(f"  {ts.student_type:<24} {before:>8.3f}  "
                  f"{ts.mean_final_knowledge:>8.3f}  "
                  f"{ts.mean_knowledge_gain:>+7.4f}")
        print()

    # ── timing summary ─────────────────────────────────────────────────────
    print(f"  Evaluation wall-clock time : {eval_time:.1f}s ({eval_time/60:.1f} min)")
    if not args.skip_training:
        total_time = training_time + eval_time
        print(f"  Total wall-clock time      : {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── save outputs ───────────────────────────────────────────────────────
    # 1) JSON report
    report_dict: Dict = summary.to_dict()
    report_dict["meta"] = {
        "timestamp": timestamp,
        "train_seed": args.train_seed,
        "test_seed": args.test_seed,
        "timesteps": args.timesteps if not args.skip_training else 0,
        "training_time_s": round(training_time, 2) if not args.skip_training else 0,
        "eval_time_s": round(eval_time, 2),
        "api_base_url": args.api_base_url,
        "problems_loaded": len(test_env.problem_bank.problems),
        "concept_dim": test_env.concept_dim,
        "action_dim": test_env.action_dim,
        "state_dim": test_env.state_dim,
        "balanced": args.balanced,
    }
    if pre_snap:
        report_dict["pre_training_snapshot"] = pre_snap

    json_path = out_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"\nSaved JSON report → {json_path}")

    # 2) Human-readable text metrics
    txt_path = out_dir / "metrics_summary.txt"
    with open(txt_path, "w") as f:
        f.write(f"PPO Tutoring Agent  -  Train & Evaluate\n")
        f.write(f"Timestamp        : {timestamp}\n")
        f.write(f"Train seed       : {args.train_seed}\n")
        f.write(f"Test seed        : {args.test_seed}\n")
        if not args.skip_training:
            f.write(f"Training steps   : {args.timesteps:,}\n")
            f.write(f"Training time    : {training_time:.1f}s\n")
        f.write(f"Eval episodes    : {args.eval_episodes}\n")
        f.write(f"Mastery threshold: {args.mastery_threshold:.0%}\n")
        f.write("\n--- Q1: Learning Effectiveness ---\n")
        f.write(f"Mean knowledge gain : {summary.mean_knowledge_gain:+.4f}\n")
        f.write(f"Mean NLG            : {summary.mean_nlg:+.4f}\n")
        f.write(f"All types improved  : {summary.all_types_improved}\n")
        f.write("\n--- Q2: Exam Readiness ---\n")
        f.write(f"Fraction exam-ready : {summary.exam_readiness.fraction_ready:.1%}\n")
        f.write(f"Mean final knowledge: {summary.exam_readiness.mean_final_knowledge:.3f}\n")
        f.write(f"Mean NLG            : {summary.exam_readiness.mean_nlg:+.4f}\n")
        f.write("\n--- Q3: Personalization ---\n")
        f.write(f"ZPD adherence       : {summary.personalization.zpd_adherence_global:.1%}\n")
        f.write(f"Diff-knowledge corr : {summary.personalization.difficulty_knowledge_correlation:+.3f}\n")
        f.write(f"Remediation success : {summary.personalization.remediation_followup_success_rate:.1%}\n")
        f.write("\n--- Q4: Engagement ---\n")
        f.write(f"Give-up rate        : {summary.give_up_rate:.1%}\n")
        f.write(f"Motivation delta    : {summary.mean_motivation_end - summary.mean_motivation_start:+.4f}\n")
        f.write(f"Mean reward         : {summary.mean_reward:.2f}\n")
        f.write("\n--- Q5: Efficiency ---\n")
        f.write(f"Success rate        : {summary.success_rate:.1%}\n")
        f.write(f"Problems to mastery : {summary.mean_problems_to_mastery:.1f}\n")
        if summary.baselines:
            f.write("\n--- Q6: Baseline Comparison ---\n")
            f.write(f"{'Policy':<26} {'Gain':>7}  {'NLG':>7}  {'Ready':>7}  {'Reward':>8}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'>>> PPO (this model) <<<':<26} "
                    f"{summary.mean_knowledge_gain:>+7.4f}  "
                    f"{summary.mean_nlg:>+7.4f}  "
                    f"{summary.exam_readiness.fraction_ready:>7.1%}  "
                    f"{summary.mean_reward:>8.2f}\n")
            for b in summary.baselines:
                f.write(f"{b.name:<26} "
                        f"{b.mean_knowledge_gain:>+7.4f}  "
                        f"{b.mean_nlg:>+7.4f}  "
                        f"{b.fraction_exam_ready:>7.1%}  "
                        f"{b.mean_reward:>8.2f}\n")
        if summary.quartile_analysis:
            f.write("\n--- Quartile Analysis ---\n")
            for q in summary.quartile_analysis.quartiles:
                r = q.initial_knowledge_range
                f.write(f"Q{q.quartile} [{r[0]:.2f},{r[1]:.2f}]: "
                        f"gain={q.mean_knowledge_gain:+.4f}  "
                        f"NLG={q.normalized_learning_gain:+.4f}  "
                        f"mastery={q.mastery_rate:.1%}\n")
            f.write(f"Low-performer advantage: {summary.quartile_analysis.low_performer_advantage:+.4f}\n")
        if summary.concept_analysis:
            f.write("\n--- Top Concept Domains ---\n")
            sorted_domains = sorted(
                summary.concept_analysis.domain_mean_gain.items(),
                key=lambda x: x[1], reverse=True,
            )
            for domain, gain in sorted_domains:
                mr = summary.concept_analysis.domain_mastery_rate.get(domain, 0.0)
                f.write(f"{domain:<22} gain={gain:+.4f}  mastery={mr:.1%}\n")

    print(f"Saved text summary  → {txt_path}")
    print(f"\nAll outputs saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()
