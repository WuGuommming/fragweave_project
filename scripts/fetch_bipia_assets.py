#!/usr/bin/env python
from __future__ import annotations

import argparse

from fragweave.data.bipia_fetch import ensure_bipia_repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=str, default="data/bipia")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    p = ensure_bipia_repo(args.dst, force=args.force)
    print(f"BIPIA assets ready at: {p}")


if __name__ == "__main__":
    main()
