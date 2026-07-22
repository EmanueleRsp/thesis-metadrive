from __future__ import annotations

import pytest

from thesis_rl.runtime.execution import seeding


def test_configure_parent_torch_threads_leaves_null_unchanged(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        seeding.torch, "set_num_threads", lambda value: calls.append(("intra", value))
    )
    monkeypatch.setattr(
        seeding.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )

    seeding.configure_parent_torch_threads(None)

    assert calls == []


def test_configure_parent_torch_threads_sets_both_pools(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        seeding.torch, "set_num_threads", lambda value: calls.append(("intra", value))
    )
    monkeypatch.setattr(
        seeding.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )

    seeding.configure_parent_torch_threads(2)

    assert calls == [("intra", 2), ("interop", 2)]


def test_configure_parent_torch_threads_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        seeding.configure_parent_torch_threads(0)
