from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


INJSQUAD_BENCHMARK_NAME = "injsquad"

_LOCAL_REFERENCE_ARCHIVE = "indirect-pia-detection-main.zip"
_LOCAL_REFERENCE_PREFIX = "indirect-pia-detection-main/data"

_RAW_DIR = Path("data/injsquad/raw")
_MANDATORY_BENCHMARK_FILE = "crafted_instruction_data_squad_injection_qa.json"
_OPTIONAL_CONTEXT_FILE = "crafted_instruction_data_context_squad.json"
_OPTIONAL_ALPACA_FILE = "crafted_instruction_data_alpaca.json"
_OPTIONAL_DAVINCI_FILE = "crafted_instruction_data_davinci.json"


@dataclass(frozen=True)
class InjSquadPaths:
    """Centralized path conventions for local Inj-SQuAD integration."""

    repo_root: Path = Path(".")

    @property
    def local_reference_archive(self) -> Path:
        return self.repo_root / _LOCAL_REFERENCE_ARCHIVE

    @property
    def local_reference_data_prefix(self) -> str:
        return _LOCAL_REFERENCE_PREFIX

    @property
    def raw_dir(self) -> Path:
        return self.repo_root / _RAW_DIR

    @property
    def squad_injection_qa_json(self) -> Path:
        return self.raw_dir / _MANDATORY_BENCHMARK_FILE

    @property
    def context_squad_json(self) -> Path:
        return self.raw_dir / _OPTIONAL_CONTEXT_FILE

    @property
    def alpaca_json(self) -> Path:
        return self.raw_dir / _OPTIONAL_ALPACA_FILE

    @property
    def davinci_json(self) -> Path:
        return self.raw_dir / _OPTIONAL_DAVINCI_FILE


def get_default_paths(repo_root: str | Path = ".") -> InjSquadPaths:
    return InjSquadPaths(repo_root=Path(repo_root))


def assert_mandatory_benchmark_exists(paths: InjSquadPaths) -> Path:
    benchmark_file = paths.squad_injection_qa_json
    if not benchmark_file.exists():
        raise FileNotFoundError(
            "Missing Inj-SQuAD benchmark file: "
            "data/injsquad/raw/crafted_instruction_data_squad_injection_qa.json"
        )
    return benchmark_file


def validate_optional_native_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing optional Inj-SQuAD native file ({label}): {path}")
    return path
