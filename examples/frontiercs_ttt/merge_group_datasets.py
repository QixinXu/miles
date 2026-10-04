#!/usr/bin/env python3
"""Merge per-source Miles group JSONL files into one mixed-source stream.

Each input row is retained once. Rows are emitted in deterministic round-robin
order across inputs, and metadata is annotated with a source-specific dataset
and namespaced group ID so independent datasets can safely reuse local IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ALIASES = {
    "frontier_cs": "frontier_cs",
    "frontier-cs": "frontier_cs",
    "fcs": "frontier_cs",
    "frontiersmith": "frontiersmith",
    "frontier_smith": "frontiersmith",
    "ale": "ale_bench",
    "ale-bench": "ale_bench",
    "ale_bench": "ale_bench",
}


def _parse_source(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("source must be DATASET=/path/to/groups.jsonl")
    normalized = ALIASES.get(name.strip().lower())
    if normalized is None:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {name!r}; choose frontier_cs, frontiersmith, or ale_bench"
        )
    return normalized, Path(path.strip()).expanduser().resolve()


def _load_rows(source: str, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {source} group file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number} has no metadata object")
            group_id = str(metadata.get("group_id") or "").strip()
            problem_ids = metadata.get("problem_ids")
            if not group_id or not isinstance(problem_ids, list) or not problem_ids:
                raise ValueError(
                    f"{path}:{line_number} must have metadata.group_id and non-empty problem_ids"
                )
            metadata = dict(metadata)
            # Preserve explicit per-problem routing for intentionally mixed
            # groups; otherwise each file's declared source is the default.
            if not metadata.get("dataset"):
                metadata["dataset"] = source
            metadata["group_id"] = f"{source}__{group_id}"
            row["metadata"] = metadata
            rows.append(row)
    if not rows:
        raise ValueError(f"{source} group file is empty: {path}")
    return rows


def merge(sources: list[tuple[str, Path]], output: Path) -> dict[str, int]:
    resolved_output = output.expanduser().resolve()
    loaded = []
    for source, path in sources:
        if path.resolve() == resolved_output:
            raise ValueError("output path must not overwrite an input source")
        loaded.append((source, _load_rows(source, path)))

    # Round-robin the source files, retaining each source's original row order.
    merged: list[dict[str, Any]] = []
    max_rows = max(len(rows) for _, rows in loaded)
    for index in range(max_rows):
        for _, rows in loaded:
            if index < len(rows):
                merged.append(rows[index])

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved_output.name}.", suffix=".tmp", dir=resolved_output.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in merged:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, resolved_output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {source: len(rows) for source, rows in loaded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=_parse_source,
        required=True,
        metavar="DATASET=JSONL",
        help="repeat once per data source; rows are interleaved round-robin",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = merge(args.source, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "groups_by_source": counts}))


if __name__ == "__main__":
    main()
