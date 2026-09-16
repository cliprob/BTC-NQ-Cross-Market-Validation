from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_frozen_research_artifacts_are_internally_consistent() -> None:
    output = Path("outputs/portfolio")
    manifest = json.loads((output / "validation_manifest.json").read_text(encoding="utf-8"))
    result_path = output / "final_results.json"
    results = json.loads(result_path.read_text(encoding="utf-8"))
    assert manifest["final_test"]["evaluated"] is True
    assert manifest["final_test"]["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
    assert results["locked_setting_hash"] == manifest["locked_setting_hash"]
    assert "C:\\Users" not in json.dumps(manifest)
    assert (output / "reliability_oof.png").stat().st_size > 0
    assert (output / "final_test_snapshot.png").stat().st_size > 0
    assert Path("report/main.pdf").stat().st_size > 100_000
