from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def load_validator() -> ModuleType:
    module_path = Path(__file__).parents[1] / "scripts" / "validate_torch_install.py"
    spec = importlib.util.spec_from_file_location("validate_torch_install", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_elf_header(path: Path, machine: int) -> None:
    header = bytearray(20)
    header[:4] = b"\x7fELF"
    header[18:20] = machine.to_bytes(2, byteorder="little")
    path.write_bytes(header)


class FakeDistribution:
    def __init__(self, root: Path) -> None:
        self.root = root

    def locate_file(self, relative_path: Path) -> Path:
        return self.root / relative_path


def test_validate_cusparselt_payload_accepts_aarch64_elf(tmp_path: Path) -> None:
    validator = load_validator()
    library = tmp_path / "nvidia/cusparselt/lib/libcusparseLt.so.0"
    library.parent.mkdir(parents=True)
    write_elf_header(library, machine=183)

    result = validator.validate_cusparselt_payload("aarch64", FakeDistribution(tmp_path))

    assert result == library


def test_validate_cusparselt_payload_rejects_wrong_elf_machine(tmp_path: Path) -> None:
    validator = load_validator()
    library = tmp_path / "nvidia/cusparselt/lib/libcusparseLt.so.0"
    library.parent.mkdir(parents=True)
    write_elf_header(library, machine=62)

    with pytest.raises(RuntimeError, match="architecture mismatch"):
        validator.validate_cusparselt_payload("aarch64", FakeDistribution(tmp_path))


def test_validate_cusparselt_payload_rejects_unknown_architecture(tmp_path: Path) -> None:
    validator = load_validator()

    with pytest.raises(RuntimeError, match="Unsupported architecture"):
        validator.validate_cusparselt_payload("ppc64le", FakeDistribution(tmp_path))
