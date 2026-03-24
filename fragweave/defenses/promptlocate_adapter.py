from __future__ import annotations

import json
import sys
from contextlib import contextmanager
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


@contextmanager
def _patch_torch_bfloat16_numpy():
    try:
        import torch
    except Exception:
        yield False
        return
    orig_numpy = torch.Tensor.numpy

    def _safe_numpy(self):
        if getattr(self, "dtype", None) == torch.bfloat16:
            tensor = self.detach().to(dtype=torch.float32).cpu()
            return orig_numpy(tensor)
        return orig_numpy(self)

    torch.Tensor.numpy = _safe_numpy
    try:
        yield True
    finally:
        torch.Tensor.numpy = orig_numpy


class OpenPromptInjectionAdapter:
    """Thin wrapper around the public Open-Prompt-Injection interfaces."""

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
        try:
            raw_value = int(detector.detect(prompt))
            return DetectorOutcome(detected=bool(raw_value), raw_value=raw_value)
        except Exception as e:
            if "BFloat16" in repr(e):
                with _patch_torch_bfloat16_numpy():
                    raw_value = int(detector.detect(prompt))
                    return DetectorOutcome(detected=bool(raw_value), raw_value=raw_value)
            raise

    def locate_and_recover(self, prompt: str, target_instruction: str) -> Tuple[str, str]:
        recovered, localized, _ = self.locate_and_recover_with_debug(prompt, target_instruction)
        return recovered, localized

    def _run_locate_impl(self, prompt: str, target_instruction: str, debug: Dict[str, Any]) -> Tuple[str, str]:
        locator = self._get_locator()
        from OpenPromptInjection.apps.PromptLocate import split_sentence, binary_search_injection, merge_intervals

        segments = split_sentence(
            prompt,
            locator.nlp,
            locator.bd.model.tokenizer,
            locator.embedding_layer,
            locator.sep_thres,
        )
        debug["segments"] = list(segments)
        injection_start_end, tot_cnt = binary_search_injection(
            segments,
            locator.bd,
            target_instruction,
            locator.helper_tokenizer,
            locator.helper_model,
        )
        debug["injection_start_end"] = [list(x) for x in injection_start_end]
        debug["total_queries"] = int(tot_cnt)
        if not injection_start_end:
            return prompt, ""

        injection_starts = [int(start) for start, _ in injection_start_end]
        injection_ends = [int(end) for _, end in injection_start_end]
        merged_starts, merged_ends = merge_intervals(injection_starts, injection_ends)
        debug["merged_intervals"] = [[int(a), int(b)] for a, b in zip(merged_starts, merged_ends)]

        recovered_text = " ".join(segments[:injection_start_end[0][0]]) + " "
        for i in range(len(injection_start_end) - 1):
            recovered_text += (
                " ".join(segments[injection_start_end[i][1]:injection_start_end[i + 1][0]]) + " "
                if injection_start_end[i][1] < injection_start_end[i + 1][0]
                else ""
            )
        recovered_text += " ".join(segments[injection_start_end[-1][1]:])
        localized_text = " ".join(segments[merged_starts[0]:merged_ends[0]])
        for i in range(1, len(merged_starts)):
            localized_text += " " + " ".join(segments[merged_starts[i]:merged_ends[i]])
        debug["localized_text"] = localized_text
        debug["recovered_text"] = recovered_text
        return recovered_text, localized_text

    def locate_and_recover_with_debug(self, prompt: str, target_instruction: str) -> Tuple[str, str, Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "segments": [],
            "injection_start_end": [],
            "merged_intervals": [],
            "total_queries": 0,
            "error": None,
            "initial_error": None,
            "retry_after_bfloat16_patch": False,
            "retry_error": None,
        }
        try:
            recovered_text, localized_text = self._run_locate_impl(prompt, target_instruction, debug)
            return recovered_text, localized_text, debug
        except Exception as e:
            debug["initial_error"] = repr(e)
            if "BFloat16" in repr(e):
                try:
                    with _patch_torch_bfloat16_numpy() as patched:
                        debug["retry_after_bfloat16_patch"] = bool(patched)
                        recovered_text, localized_text = self._run_locate_impl(prompt, target_instruction, debug)
                        return recovered_text, localized_text, debug
                except Exception as e2:
                    debug["retry_error"] = repr(e2)
                    debug["error"] = repr(e2)
                    return prompt, "", debug
            debug["error"] = repr(e)
            return prompt, "", debug
