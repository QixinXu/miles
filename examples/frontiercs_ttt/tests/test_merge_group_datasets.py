from __future__ import annotations

import json

from examples.frontiercs_ttt.merge_group_datasets import merge


def _write_groups(path, group_ids):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "group"}],
                    "metadata": {"group_id": group_id, "problem_ids": [group_id]},
                }
            )
            + "\n"
            for group_id in group_ids
        ),
        encoding="utf-8",
    )


def test_merge_interleaves_sources_and_adds_routing_metadata(tmp_path):
    fcs = tmp_path / "fcs.jsonl"
    smith = tmp_path / "smith.jsonl"
    ale = tmp_path / "ale.jsonl"
    _write_groups(fcs, ["g0", "g1"])
    _write_groups(smith, ["g0"])
    _write_groups(ale, ["g0", "g1", "g2"])
    output = tmp_path / "merged.jsonl"

    counts = merge(
        [("frontier_cs", fcs), ("frontiersmith", smith), ("ale_bench", ale)],
        output,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert counts == {"frontier_cs": 2, "frontiersmith": 1, "ale_bench": 3}
    assert [row["metadata"]["dataset"] for row in rows] == [
        "frontier_cs",
        "frontiersmith",
        "ale_bench",
        "frontier_cs",
        "ale_bench",
        "ale_bench",
    ]
    assert len({row["metadata"]["group_id"] for row in rows}) == len(rows)
    assert rows[0]["metadata"]["group_id"] == "frontier_cs__g0"


def test_merge_preserves_per_problem_source_overrides(tmp_path):
    source = tmp_path / "mixed.jsonl"
    source.write_text(
        json.dumps(
            {
                "prompt": [],
                "metadata": {
                    "group_id": "mixed",
                    "problem_ids": ["174", "ahc001"],
                    "problem_datasets": {"ahc001": "ale_bench"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "merged.jsonl"

    merge([("frontier_cs", source)], output)

    metadata = json.loads(output.read_text())["metadata"]
    assert metadata["dataset"] == "frontier_cs"
    assert metadata["problem_datasets"] == {"ahc001": "ale_bench"}
