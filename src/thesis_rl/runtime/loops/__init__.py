"""Runtime training/evaluation loops."""

from thesis_rl.runtime.loops.eval_loop import run_evaluation
from thesis_rl.runtime.loops.train_loop import run_training

__all__ = ["run_evaluation", "run_training"]
