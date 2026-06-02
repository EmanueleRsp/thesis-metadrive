from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class CurriculumState:
    stage_index: int = 0
    stage_steps_done: int = 0
    eval_count_at_stage: int = 0
    consecutive_passes: int = 0
    last_eval_passed: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CurriculumState":
        if data is None:
            return cls()
        return cls(
            stage_index=int(data.get("stage_index", 0)),
            stage_steps_done=int(data.get("stage_steps_done", 0)),
            eval_count_at_stage=int(data.get("eval_count_at_stage", 0)),
            consecutive_passes=int(data.get("consecutive_passes", 0)),
            last_eval_passed=bool(data.get("last_eval_passed", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": int(self.stage_index),
            "stage_steps_done": int(self.stage_steps_done),
            "eval_count_at_stage": int(self.eval_count_at_stage),
            "consecutive_passes": int(self.consecutive_passes),
            "last_eval_passed": bool(self.last_eval_passed),
        }
