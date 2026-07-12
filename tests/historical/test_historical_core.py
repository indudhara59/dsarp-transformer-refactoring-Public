from dsarp.arcan.smell_matcher import MatchConfig, jaccard, match_smells
from dsarp.historical.label_confidence import label_confidence
from dsarp.historical.schemas import validate_inference_features
from dsarp.mining.refactoring_parser import parse_refactoringminer


def test_jaccard_and_exact_match():
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    old = {"repository": "r", "smell_type": "godComponent", "central_component": "a.B", "affected_component_name": "a.C", "smell_id": "1"}
    new = {**old, "smell_id": "99"}
    matched = match_smells([old], [new], MatchConfig(threshold=.5))
    assert matched[0]["after"]["smell_id"] == "99"
    assert matched[0]["transition"] == "persisted"


def test_resolved_and_new_transitions():
    old = {"repository": "r", "smell_type": "unstableDep", "central_component": "a", "smell_id": "1"}
    new = {"repository": "r", "smell_type": "hubLikeDep", "central_component": "b", "smell_id": "2"}
    transitions = match_smells([old], [new])
    assert {item["transition"] for item in transitions} == {"resolved", "newly_introduced"}


def test_refactoring_parser_preserves_sides():
    payload = {"commits": [{"refactorings": [{"type": "Move Class", "leftSideLocations": [{"filePath": "A.java", "codeElement": "p.A"}], "rightSideLocations": [{"filePath": "B.java", "codeElement": "q.A"}]}]}]}
    row = parse_refactoringminer(payload)[0]
    assert row["refactoring_id"] == "move_class"
    assert row["source_elements"] == ["p.A"] and row["target_elements"] == ["q.A"]


def test_inference_manifest_rejects_future_fields():
    validate_inference_features(["severity", "fan_out", "historical_frequency"])
    try: validate_inference_features(["severity_after", "atdi_delta"])
    except ValueError: pass
    else: raise AssertionError("leakage was accepted")


def test_ambiguity_penalty_can_reject_label():
    cfg = {"weights": {"element_overlap": .5, "ambiguity_penalty": .8, "unrelated_refactoring_penalty": .2}, "quality_bands": {"strong_historical_label": .7, "medium_historical_label": .5, "weak_historical_label": .25}}
    result = label_confidence({"element_overlap": 1, "ambiguity_penalty": 1}, cfg)
    assert result["quality_band"] == "rejected_historical_association"
