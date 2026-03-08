#!/usr/bin/env python3
"""Lightweight preflight checks before claiming PR/commit has been published.

This script is intentionally static/offline and only inspects local git metadata.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return out.strip()


def check_remote_exists() -> CheckResult:
    remotes = _run(["git", "remote"]).splitlines()
    if remotes:
        return CheckResult("remote_configured", True, f"remotes={','.join(remotes)}")
    return CheckResult("remote_configured", False, "no git remote configured")


def check_upstream() -> CheckResult:
    try:
        upstream = _run(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    except subprocess.CalledProcessError:
        return CheckResult("upstream_tracking", False, "current branch has no upstream tracking branch")
    return CheckResult("upstream_tracking", True, f"upstream={upstream}")


def check_clean_worktree() -> CheckResult:
    status = _run(["git", "status", "--short"])
    if not status:
        return CheckResult("worktree_clean", True, "clean")
    return CheckResult("worktree_clean", False, "worktree has uncommitted changes")


def main() -> int:
    checks = [check_remote_exists(), check_upstream(), check_clean_worktree()]
    failed = [c for c in checks if not c.ok]

    for c in checks:
        flag = "PASS" if c.ok else "FAIL"
        print(f"[{flag}] {c.name}: {c.detail}")

    if failed:
        print("\nPreflight failed: do not claim push/PR exists on GitHub.")
        return 1

    print("\nPreflight passed: repository is configured for push/PR workflow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
