"""Scoped pre-spawn limits for numerical-library worker pools."""

from __future__ import annotations

import os
from typing import Any


NUMERIC_WORKER_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def start_process_with_numeric_thread_limit(
    process: Any,
    numeric_library_num_threads: int | None,
) -> None:
    """Start a child with an optional numeric-pool cap inherited before imports.

    Spawned children import NumPy and extension modules before entering their
    application target. The parent environment is restored immediately after
    ``start()`` returns, so the cap is local to the newly created child.
    """

    if numeric_library_num_threads is None:
        process.start()
        return
    if numeric_library_num_threads <= 0:
        raise ValueError("Worker numeric-library thread count must be positive")

    previous = {key: os.environ.get(key) for key in NUMERIC_WORKER_THREAD_ENV_KEYS}
    try:
        value = str(int(numeric_library_num_threads))
        for key in NUMERIC_WORKER_THREAD_ENV_KEYS:
            os.environ[key] = value
        process.start()
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
