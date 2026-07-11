from pathlib import Path

import pandas as pd
import pytest

from dsarp.baseline.diversity import diverse_top_k
from dsarp.baseline.ranker import deterministic_sort
from dsarp.candidates.generator import generate_candidates, stable_candidate_id
from dsarp.ingestion.file_discovery import discover_files
from dsarp.ingestion.joiner import join_arcan
from dsarp.preprocessing.filtering import filter_selected_smells, normalize_smell_type
from dsarp.preprocessing.missing_values import parse_numeric


@pytest.mark.parametrize(("raw", "expected"), [("God Component", "godComponent"), ("god_component", "godComponent"), ("UNSTABLE DEPENDENCY", "unstableDep"), ("hubLikeDependency", "hubLikeDep"), ("cyclicDep", None)])
def test_smell_aliases(raw, expected):
    assert normalize_smell_type(raw) == expected


def test_discovery_nested(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    for name in ("affects-smell-affects.csv", "metrics-component-metrics.csv", "smells-smell-characteristics.csv"):
        (tmp_path / "raw" / name).write_text("x\n")
    assert set(discover_files(tmp_path)) == {"affects", "metrics", "smells"}


def test_filtering_stats():
    selected, stats = filter_selected_smells(pd.DataFrame({"raw_smell_type": ["godComponent", "cyclicDep"]}))
    assert len(selected) == 1 and stats["excluded"] == 1


def test_numeric_parsing():
    result = parse_numeric(pd.Series(["1,200", " 3.5 ", None]))
    assert result.iloc[:2].tolist() == [1200.0, 3.5] and pd.isna(result.iloc[2])


def test_composite_join_and_unmatched_reporting():
    smells = pd.DataFrame([{"project": "p", "version_id": "v", "smell_id": "s", "smell_type": "godComponent"}])
    affects = pd.DataFrame([{"project": "p", "version_id": "v", "smell_id": "s", "affected_component_id": "c", "affected_component_name": "C"}])
    metrics = pd.DataFrame([{"project": "p", "version_id": "v", "affected_component_id": "c", "metric_component_name": "C", "fan_in": 1}])
    result, quality = join_arcan(smells, affects, metrics)
    assert len(result) == 1 and quality["unmatched_affected_components"] == 0


def test_duplicate_metric_key_rejected():
    smells = pd.DataFrame([{"project": "p", "version_id": "v", "smell_id": "s"}]); affects = pd.DataFrame([{"project": "p", "version_id": "v", "smell_id": "s", "affected_component_id": "c"}]); metric = {"project": "p", "version_id": "v", "affected_component_id": "c"}
    with pytest.raises(Exception): join_arcan(smells, affects, pd.DataFrame([metric, metric]))


def test_candidate_ids_and_count():
    row = pd.Series({"project": "p", "version_id": "v", "smell_id": "s", "affected_component_id": "c", "smell_type": "godComponent"})
    taxonomy = {"split": {"smell_types": ["godComponent"], "display_name": "Split", "description": "d", "intended_effect": "e", "implementation_outline": "i", "warning": "w", "family": "f", "applicability_prior": .5, "benefit_prior": .5, "risk_prior": .5}}
    assert stable_candidate_id(row, "split") == stable_candidate_id(row, "split")
    assert len(generate_candidates(pd.DataFrame([row]), taxonomy)) == 1


def test_ranking_tie_breaks():
    frame = pd.DataFrame([{"candidate_id": "b", "final_priority_score": .5, "expected_benefit": .5, "estimated_risk": .2}, {"candidate_id": "a", "final_priority_score": .5, "expected_benefit": .5, "estimated_risk": .2}])
    assert deterministic_sort(frame).candidate_id.tolist() == ["a", "b"]


def test_diversity_constraints():
    rows = [{"candidate_id": str(i), "smell_id": "s", "affected_component_id": "c", "recommendation_family": str(i), "final_priority_score": 1-i/10, "expected_benefit": .5, "estimated_risk": .2, "pressure_score": .5, "centrality_composite": .5, "size_composite": .5} for i in range(4)]
    result = diverse_top_k(pd.DataFrame(rows), 4, {"max_per_smell": 2, "max_per_component": 3, "mmr_lambda": .8, "min_distinct_smell_types": 0})
    assert len(result) == 2
