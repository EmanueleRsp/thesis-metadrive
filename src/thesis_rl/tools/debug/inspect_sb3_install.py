from __future__ import annotations

import json
from pathlib import Path
from importlib import metadata


def main() -> None:
    import stable_baselines3 as sb3

    sb3_path = Path(sb3.__file__).resolve()
    print(f"stable_baselines3.__file__={sb3_path}")
    print(f"stable_baselines3.__version__={getattr(sb3, '__version__', 'unknown')}")
    try:
        dist = metadata.distribution("stable-baselines3")
        direct_url = dist.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct_url = None
    if direct_url:
        try:
            payload = json.loads(direct_url)
        except json.JSONDecodeError:
            payload = {"raw": direct_url}
        print(f"stable_baselines3.direct_url={payload}")


if __name__ == "__main__":
    main()
