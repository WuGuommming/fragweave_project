from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class OpenPIPaths:
    repo_root: Path
    model_config_json: Path
    detector_ft_path: Path
    promptlocate_ft_path: Path


@dataclass
class DetectorOutcome:
    detected: bool
    raw_value: int


class OpenPromptInjectionAssetsError(RuntimeError):
    pass


class OpenPromptInjectionAdapter:
    """Thin wrapper around the public Open-Prompt-Injection interfaces.

    This class intentionally follows the upstream usage pattern shown in the
    README:
      - DataSentinelDetector(config).detect(prompt)
      - PromptLocate(config).locate_and_recover(prompt, target_instruction)

    We keep detector and PromptLocate as separate objects because the upstream
    PromptLocate constructor internally builds its own detector from the config,
    and the README uses a different ft_path for PromptLocate than for the
    standalone DataSentinel detector.
    """

    def __init__(
        self,
        *,
        repo_root: str | Path,
        model_config_json: str | Path,
        detector_ft_path: str | Path,
        promptlocate_ft_path: str | Path,
        helper_model_name: str = "gpt2",
        sep_thres: float = 0.0,
    ) -> None:
        self.paths = OpenPIPaths(
            repo_root=Path(repo_root),
            model_config_json=Path(model_config_json),
            detector_ft_path=Path(detector_ft_path),
            promptlocate_ft_path=Path(promptlocate_ft_path),
        )
        self.helper_model_name = helper_model_name
        self.sep_thres = float(sep_thres)
        self._validate_paths()
        self._prepare_import_path()
        self._detector = None
        self._locator = None

    def _validate_paths(self) -> None:
        missing = []
        if not self.paths.repo_root.exists():
            missing.append(f"repo_root={self.paths.repo_root}")
        if not self.paths.model_config_json.exists():
            missing.append(f"model_config_json={self.paths.model_config_json}")
        if not self.paths.detector_ft_path.exists():
            missing.append(f"detector_ft_path={self.paths.detector_ft_path}")
        if not self.paths.promptlocate_ft_path.exists():
            missing.append(f"promptlocate_ft_path={self.paths.promptlocate_ft_path}")
        if missing:
            raise OpenPromptInjectionAssetsError(
                "Missing Open-Prompt-Injection assets: " + ", ".join(missing)
            )

    def _prepare_import_path(self) -> None:
        repo_root = str(self.paths.repo_root.resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _load_config(self, ft_path: Path) -> Dict[str, Any]:
        obj = json.loads(self.paths.model_config_json.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise OpenPromptInjectionAssetsError(
                f"Invalid model config JSON: {self.paths.model_config_json}"
            )
        params = obj.setdefault("params", {})
        if not isinstance(params, dict):
            raise OpenPromptInjectionAssetsError(
                f"Invalid params in model config JSON: {self.paths.model_config_json}"
            )
        params["ft_path"] = str(ft_path)
        return obj

    def _get_detector(self):
        if self._detector is None:
            from OpenPromptInjection.apps.DataSentinelDetector import DataSentinelDetector

            cfg = self._load_config(self.paths.detector_ft_path)
            self._detector = DataSentinelDetector(cfg)
        return self._detector

    def _get_locator(self):
        if self._locator is None:
            from OpenPromptInjection.apps.PromptLocate import PromptLocate

            cfg = self._load_config(self.paths.promptlocate_ft_path)
            self._locator = PromptLocate(cfg)
        return self._locator

    def detect(self, prompt: str) -> DetectorOutcome:
        detector = self._get_detector()
        raw_value = int(detector.detect(prompt))
        return DetectorOutcome(detected=bool(raw_value), raw_value=raw_value)

    def locate_and_recover(self, prompt: str, target_instruction: str) -> Tuple[str, str]:
        locator = self._get_locator()
        recovered_prompt, localized_prompt = locator.locate_and_recover(
            prompt, target_instruction
        )
        return str(recovered_prompt), str(localized_prompt)
