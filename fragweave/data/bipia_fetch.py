from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import urllib.request


BIPIA_GIT = "https://github.com/microsoft/BIPIA.git"
BIPIA_ZIP = "https://codeload.github.com/microsoft/BIPIA/zip/refs/heads/main"


def ensure_bipia_repo(dst_root: str | Path, *, force: bool = False) -> Path:
    """Ensure BIPIA repo assets exist locally.

    Tries, in order:
    1) reuse existing directory
    2) git clone (requires git and internet)
    3) download zip from GitHub and extract

    Returns the resolved path to the extracted/checked-out repo root.
    """

    dst_root = Path(dst_root).expanduser().resolve()
    if force and dst_root.exists():
        shutil.rmtree(dst_root)

    if dst_root.exists() and (dst_root / "benchmark").exists():
        return dst_root

    dst_root.parent.mkdir(parents=True, exist_ok=True)

    # 1) git clone
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", BIPIA_GIT, str(dst_root)],
            check=True,
        )
        if (dst_root / "benchmark").exists():
            return dst_root
    except Exception:
        pass

    # 2) download zip
    tmp_zip = dst_root.parent / "BIPIA_main.zip"
    try:
        urllib.request.urlretrieve(BIPIA_ZIP, tmp_zip)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(dst_root.parent)
        # extracted folder is typically BIPIA-main
        extracted = dst_root.parent / "BIPIA-main"
        if extracted.exists():
            if dst_root.exists():
                shutil.rmtree(dst_root)
            extracted.rename(dst_root)
        if (dst_root / "benchmark").exists():
            return dst_root
    finally:
        if tmp_zip.exists():
            try:
                tmp_zip.unlink()
            except Exception:
                pass

    raise RuntimeError(
        f"Failed to obtain BIPIA assets. Tried git clone and zip download. "
        f"Please manually clone {BIPIA_GIT} into {dst_root} and retry."
    )


def ensure_bipia_benchmark_jsonl(bipia_root: str | Path, task: str) -> None:
    """Ensure BIPIA benchmark jsonl exists for license-gated tasks.

    Official BIPIA repo does not ship WebQA (benchmark/qa) or Summarization
    (benchmark/abstract) jsonl files. Users must generate them.

    This helper follows the official scripts' logic, but runs non-interactively
    and caches the generated train/test.jsonl in-place for future runs.

    - WebQA: requires NewsQA files prepared per the official NewsQA repo.
      If train/test jsonl are missing, we look for the NewsQA raw files in:
        1) benchmark/qa/newsqa_data/  (recommended)
        2) benchmark/qa/newsqa/       (alternate)
        3) environment variable BIPIA_NEWSQA_DIR
      The directory must contain combined-newsqa-data-v1.csv (and optionally the .json).
    - Summarization: downloads/uses XSum via HuggingFace datasets (requires internet).
    """

    import os

    t = (task or "").lower()
    root = Path(bipia_root)
    bench = root / "benchmark"
    if not bench.exists():
        return

    # WebQA
    if t in {"web_qa", "webqa", "qa", "web"}:
        qa_dir = bench / "qa"
        if (qa_dir / "train.jsonl").exists() and (qa_dir / "test.jsonl").exists():
            return
        # Prefer a simple, repo-local convention: place NewsQA files under benchmark/qa/newsqa_data/
        # (or benchmark/qa/newsqa/) so users don't have to set env vars.
        candidates = [qa_dir / "newsqa_data", qa_dir / "newsqa"]
        newsqa_dir: Optional[Path] = None
        for c in candidates:
            if (c / "combined-newsqa-data-v1.csv").exists():
                newsqa_dir = c
                break
        if newsqa_dir is None:
            env_dir = os.environ.get("BIPIA_NEWSQA_DIR", "").strip()
            if env_dir:
                newsqa_dir = Path(env_dir)

        if newsqa_dir is None:
            raise FileNotFoundError(
                "Missing WebQA jsonl under benchmark/qa/. To auto-generate per the official BIPIA README, "
                "please download the NewsQA raw CSV (combined-newsqa-data-v1.csv) and place it under either:\n"
                "  - data/bipia/benchmark/qa/newsqa_data/\n"
                "  - data/bipia/benchmark/qa/newsqa/\n"
                "(relative to the BIPIA benchmark root), then rerun.\n"
                "Alternatively, set environment variable BIPIA_NEWSQA_DIR to that directory."
            )

        _generate_webqa_jsonl(qa_dir=qa_dir, newsqa_dir=newsqa_dir)
        return

    # Summarization
    if t in {"summarization", "summary", "abstract"}:
        abs_dir = bench / "abstract"
        if (abs_dir / "train.jsonl").exists() and (abs_dir / "test.jsonl").exists():
            return
        _generate_summarization_jsonl(abs_dir=abs_dir)
        return


def _generate_webqa_jsonl(*, qa_dir: Path, newsqa_dir: Path) -> None:
    import ast
    import csv
    import json
    import re
    import string
    from itertools import chain

    def merge_newlines(s: str) -> str:
        return re.sub(r"\n{3,}", "\n\n", s)

    def merge_lines_with_spaces(s: str) -> str:
        return re.sub(r"\n\s*\n", "\n\n", s)

    def _parse_answer_char_ranges(raw: str) -> str:
        """Match the official process.py behavior without relying on HF datasets scripts.

        The official script does: eval(example["answer_char_ranges"])[0].
        In the raw CSV, this field may already be a plain string (e.g., '196:228|None'),
        or a Python-literal list string (e.g., "['196:228|None']").
        """
        if raw is None:
            return ""
        s = str(raw)
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list) and v:
                return str(v[0])
            return str(v)
        except Exception:
            return s

    def extract_answers(row):
        story = row["story_text"]
        char_ranges = _parse_answer_char_ranges(row.get("answer_char_ranges", ""))
        char_ranges = [i.split("|") for i in char_ranges.split(",")]
        answers = []
        for cr in chain(*char_ranges):
            if cr != "None":
                start, end = map(int, cr.split(":"))
                ans = story[start:end].strip(string.punctuation + string.whitespace)
                answers.append(ans)
        row["answers"] = sorted(set(answers))
        row["story_text"] = merge_newlines(merge_lines_with_spaces(row["story_text"]))
        return row

    qa_dir.mkdir(parents=True, exist_ok=True)

    with open(qa_dir / "index.json", "r", encoding="utf-8") as f:
        indexes = json.load(f)

    csv_path = newsqa_dir / "combined-newsqa-data-v1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NewsQA raw CSV not found at {csv_path}. Expected a file named 'combined-newsqa-data-v1.csv' "
            "in the NewsQA directory."
        )

    train_idx = set(indexes.get("train", []))
    test_idx = set(indexes.get("test", []))

    train_objs = []
    test_objs = []

    # Official script selects from NewsQA 'train' split for BOTH train and test using indexes.
    # That corresponds to enumerating rows in the combined CSV in order.
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i not in train_idx and i not in test_idx:
                continue
            row2 = {
                "story_text": row.get("story_text", ""),
                "question": row.get("question", ""),
                "answer_char_ranges": row.get("answer_char_ranges", ""),
            }
            row2 = extract_answers(row2)
            obj = {"ideal": row2["answers"], "context": row2["story_text"], "question": row2["question"]}
            if i in train_idx:
                train_objs.append(obj)
            if i in test_idx:
                test_objs.append(obj)

    # Match the official script's one-off correction.
    if len(test_objs) > 87:
        test_objs[87]["ideal"] = ["Janine Sligar"]

    _write_jsonl(qa_dir / "train.jsonl", train_objs)
    _write_jsonl(qa_dir / "test.jsonl", test_objs)
    _verify_md5(qa_dir)


def _generate_summarization_jsonl(*, abs_dir: Path) -> None:
    import json

    from datasets import load_dataset

    abs_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("xsum")
    with open(abs_dir / "index.json", "r", encoding="utf-8") as f:
        indexes = json.load(f)

    train_ds = ds["train"].select(indexes["train"])
    test_ds = ds["test"].select(indexes["test"])

    train_objs = [{"ideal": s["summary"], "context": s["document"]} for s in train_ds]
    test_objs = [{"ideal": s["summary"], "context": s["document"]} for s in test_ds]

    _write_jsonl(abs_dir / "train.jsonl", train_objs)
    _write_jsonl(abs_dir / "test.jsonl", test_objs)
    _verify_md5(abs_dir)


def _write_jsonl(path: Path, rows) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _verify_md5(task_dir: Path) -> None:
    import hashlib

    md5_values = {}
    for fname in ["train.jsonl", "test.jsonl"]:
        p = task_dir / fname
        h = hashlib.md5()
        with open(p, "rb") as f:
            h.update(f.read())
        md5_values[fname] = h.hexdigest()

    md5_file = task_dir / "md5.txt"
    if not md5_file.exists():
        return
    with open(md5_file, "r", encoding="utf-8") as f:
        for line in f:
            old, name = line.strip().split("  ")
            if name in md5_values and md5_values[name] != old:
                raise ValueError(f"{task_dir.name}/{name} md5 mismatch (generated {md5_values[name]} vs expected {old})")
