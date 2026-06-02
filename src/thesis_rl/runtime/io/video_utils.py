from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] >= 3:
        return arr[..., :3]
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def render_topdown_frame(env: Any, topdown_cfg: Any) -> Any:
    kwargs = {
        "window": bool(topdown_cfg.get("window", False)),
        "screen_record": bool(topdown_cfg.get("screen_record", False)),
        "screen_size": tuple(topdown_cfg.get("screen_size", [800, 800])),
        "scaling": float(topdown_cfg.get("scaling", 4)),
        "semantic_map": bool(topdown_cfg.get("semantic_map", False)),
    }
    base_env = getattr(env, "unwrapped", env)
    try:
        return base_env.render(mode="topdown", **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'mode'" in str(exc):
            return base_env.render(**kwargs)
        raise


def save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for GIF export. Install `pillow`.") from exc

    if not frames:
        raise ValueError("No frames to save.")
    pil_frames = [Image.fromarray(to_uint8_rgb(frame)) for frame in frames]
    duration_ms = int(max(1, round(1000.0 / max(int(fps), 1))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
