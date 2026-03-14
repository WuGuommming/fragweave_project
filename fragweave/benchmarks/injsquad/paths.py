from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile


INJSQUAD_BENCHMARK_NAME = "injsquad"

_LOCAL_REFERENCE_ARCHIVE = "indirect-pia-detection-main.zip"
_LOCAL_REFERENCE_PREFIX = "indirect-pia-detection-main/data"

_RAW_DIR = Path("data/injsquad/raw")
_MANDATORY_BENCHMARK_FILE = "crafted_instruction_data_squad_injection_qa.json"
_OPTIONAL_CONTEXT_FILE = "crafted_instruction_data_context_squad.json"
_OPTIONAL_ALPACA_FILE = "crafted_instruction_data_alpaca.json"
_OPTIONAL_DAVINCI_FILE = "crafted_instruction_data_davinci.json"
_ARCHIVE_SQUAD_MEMBER = f"{_LOCAL_REFERENCE_PREFIX}/{_MANDATORY_BENCHMARK_FILE}"


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


def provision_squad_file_from_local_archive(paths: InjSquadPaths, *, overwrite: bool = False) -> Path:
    """Copy required Inj-SQuAD benchmark data from the local archive into data/injsquad/raw/."""
    dst = paths.squad_injection_qa_json
    if dst.exists() and not overwrite:
        return dst

    archive_path = paths.local_reference_archive
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing local Inj-SQuAD archive: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as zf:
        try:
            payload = zf.read(_ARCHIVE_SQUAD_MEMBER)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Missing expected member in local archive: {_ARCHIVE_SQUAD_MEMBER}"
            ) from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(payload)
    return dst


def ensure_optional_davinci_file(paths: InjSquadPaths) -> Path:
    davinci = paths.davinci_json
    if not davinci.exists():
        raise FileNotFoundError(
            "Missing optional native protocol file: data/injsquad/raw/crafted_instruction_data_davinci.json\n"
            "This file is referenced by the bundled indirect-pia-detection README but is not required for the "
            "current Inj-SQuAD integration. Place it manually only if a specific optional native sub-protocol requires it."
        )
    return davinci


def provision_native_squad_reference_files(paths: InjSquadPaths) -> dict[str, Path]:
    """Provision SQuAD-side reference files from the bundled local archive into data/injsquad/raw/."""
    archive_path = paths.local_reference_archive
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing local Inj-SQuAD archive: {archive_path}")

    members = {
        "squad_injection_qa": _ARCHIVE_SQUAD_MEMBER,
        "context_squad": f"{_LOCAL_REFERENCE_PREFIX}/{_OPTIONAL_CONTEXT_FILE}",
        "alpaca": f"{_LOCAL_REFERENCE_PREFIX}/{_OPTIONAL_ALPACA_FILE}",
    }
    out_paths = {
        "squad_injection_qa": paths.squad_injection_qa_json,
        "context_squad": paths.context_squad_json,
        "alpaca": paths.alpaca_json,
    }

    with zipfile.ZipFile(archive_path, "r") as zf:
        available = set(zf.namelist())
        for key, member in members.items():
            if member not in available:
                continue
            out = out_paths[key]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(member))

    return {k: v for k, v in out_paths.items() if v.exists()}
