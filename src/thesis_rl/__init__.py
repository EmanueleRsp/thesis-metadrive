"""thesis_rl package."""

from __future__ import annotations

import warnings


# Silence only the third-party setuptools/pkg_resources deprecation noise that
# currently bubbles up during imports. Keep all other warnings visible.
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)
