from pathlib import Path

from dsarp.pipeline.stage1_pipeline import run_stage1


def test_fixture_pipeline(tmp_path: Path):
    data = Path(__file__).parents[1] / "fixtures"
    outputs = tmp_path / "outputs"
    paths = run_stage1(data, outputs, top_k=5, config_dir=Path("configs"))
    assert all(path.exists() for path in paths.values())
    assert paths["html"].read_text().startswith("<!doctype html>")
