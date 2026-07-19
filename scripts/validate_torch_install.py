#!/usr/bin/env python3
"""Validate the installed Torch version and native cuSPARSELt payload architecture."""

from __future__ import annotations

import argparse
import platform
from importlib.metadata import Distribution, distribution
from pathlib import Path

ELF_MACHINE_BY_ARCHITECTURE = {
    "aarch64": 183,
    "x86_64": 62,
}
_ELF_MACHINE_OFFSET = 18
_ELF_HEADER_SIZE = 20
_CUSPARSELT_LIBRARY_SUFFIX = Path("nvidia/cusparselt/lib/libcusparseLt.so.0")


def expected_elf_machine(architecture: str) -> int:
    """Return the ELF e_machine identifier required by a supported architecture."""
    try:
        return ELF_MACHINE_BY_ARCHITECTURE[architecture]
    except KeyError as error:
        raise RuntimeError(
            f"Unsupported architecture for CUDA validation: {architecture}"
        ) from error


def read_elf_machine(library_path: Path) -> int:
    """Read the little-endian ELF e_machine field from a native shared library."""
    header = library_path.read_bytes()[:_ELF_HEADER_SIZE]
    if len(header) < _ELF_HEADER_SIZE or header[:4] != b"\x7fELF":
        raise RuntimeError(f"Invalid ELF library: {library_path}")
    return int.from_bytes(header[_ELF_MACHINE_OFFSET:_ELF_HEADER_SIZE], byteorder="little")


def cusparselt_library_path(package_distribution: Distribution) -> Path:
    """Return the installed cuSPARSELt shared-library path."""
    library_path = Path(package_distribution.locate_file(_CUSPARSELT_LIBRARY_SUFFIX))
    if not library_path.is_file():
        raise RuntimeError(f"Missing cuSPARSELt shared library: {library_path}")
    return library_path


def validate_cusparselt_payload(
    architecture: str,
    package_distribution: Distribution,
) -> Path:
    """Verify that the installed cuSPARSELt native payload matches the host architecture."""
    expected_machine = expected_elf_machine(architecture)
    library_path = cusparselt_library_path(package_distribution)
    actual_machine = read_elf_machine(library_path)
    if actual_machine != expected_machine:
        raise RuntimeError(
            "cuSPARSELt library architecture mismatch: "
            f"expected ELF e_machine={expected_machine}, got {actual_machine} at {library_path}"
        )
    return library_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-version", required=True, help="Expected public Torch version")
    parser.add_argument(
        "--require-cusparselt",
        action="store_true",
        help="Require and validate the CUDA cuSPARSELt native payload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    installed_version = torch.__version__.split("+", maxsplit=1)[0]
    if installed_version != args.torch_version:
        raise RuntimeError(f"Expected torch=={args.torch_version}, got {torch.__version__}")

    if args.require_cusparselt:
        architecture = platform.machine().lower()
        library_path = validate_cusparselt_payload(
            architecture=architecture,
            package_distribution=distribution("nvidia-cusparselt-cu12"),
        )
        print(f"torch={torch.__version__} cuSPARSELt={library_path} architecture={architecture}")
    else:
        print(f"torch={torch.__version__}")


if __name__ == "__main__":
    main()
