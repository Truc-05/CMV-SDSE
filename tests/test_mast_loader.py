import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascade.mast_loader import (
    MODE_IDS,
    MastSchemaError,
    analyse_label_noise,
    describe,
    is_error_for,
    load_mast_items,
    render_prompt,
    system_prompt_for,
    trajectory_of,
)


def annotation(flagged=()):
    return {m: (1 if m in flagged else 0) for m in MODE_IDS}


def full_record(trace_id, mas="ChatDev", bench="ProgramDev", llm="GPT-4o", flagged=()):
    return {
        "mas_name": mas,
        "llm_name": llm,
        "benchmark_name": bench,
        "trace_id": trace_id,
        "trace": {"index": trace_id, "key": f"{mas}_{bench}_{llm}",
                  "trajectory": f"[INFO] transcript body for trace {trace_id}"},
        "mast_annotation": annotation(flagged),
    }


@pytest.fixture
def full_path(tmp_path):
    records = (
        [full_record(i, flagged=("2.4",)) for i in range(10)]
        + [full_record(100 + i, flagged=("3.2",)) for i in range(8)]
        + [full_record(200 + i, mas="MetaGPT", bench="GAIA") for i in range(12)]
    )
    path = tmp_path / "MAD_full_dataset.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


@pytest.fixture
def human_path(tmp_path):
    records = [
        {
            "round": "Round 1",
            "mas_name": "ChatDev",
            "benchmark_name": "ProgramDev",
            "trace_id": i,
            "trace": f"transcript {i}",
            "annotations": [
                {"annotator_1": True, "annotator_2": True, "annotator_3": True,
                 "failure mode": "1.3 Step Repetition"},
                {"annotator_1": True, "annotator_2": False, "annotator_3": True,
                 "failure mode": "2.4 Information Witholding"},
                {"annotator_1": False, "annotator_2": False, "annotator_3": False,
                 "failure mode": "3.2 No or Incomplete Verification"},
            ],
        }
        for i in range(5)
    ]
    path = tmp_path / "MAD_human_labelled_dataset.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def test_target_any_vs_specific_mode():
    ann = annotation(("2.4",))
    assert is_error_for(ann, "any")
    assert is_error_for(ann, "2.4")
    assert not is_error_for(ann, "3.2")
    assert not is_error_for(ann, "fc3")
    assert is_error_for(annotation(("3.1",)), "fc3")
    assert not is_error_for(annotation(), "any")


def test_unknown_target_rejected():
    with pytest.raises(ValueError):
        is_error_for(annotation(), "9.9")


def test_trajectory_handles_both_shapes():
    assert trajectory_of({"trace": {"trajectory": "abc"}}) == "abc"
    assert trajectory_of({"trace": "plain"}) == "plain"


def test_prompt_truncates_long_transcripts():
    record = full_record(1)
    record["trace"]["trajectory"] = "x" * 5000
    prompt = render_prompt(record, max_chars=1000)
    assert "were removed to fit the context window" in prompt
    assert len(prompt) < 2000


def test_load_items_balances_and_labels(full_path):
    items = load_mast_items(full_path, target="any", n_errors=5, n_non_errors=4)
    assert len(items) == 9
    assert sum(1 for i in items if i.ground_truth_is_error) == 5
    assert all(i.payload["prompt"] for i in items)
    assert all(i.failure_mode_id == "ANY" for i in items)
    assert all(i.source_config["label_source"] == "mast_llm_annotator" for i in items)
    assert all(i.source_config["was_truncated"] is False for i in items)


def test_load_items_filters_by_mas(full_path):
    items = load_mast_items(full_path, target="any", mas_names=["MetaGPT"])
    assert items
    assert all(i.source_config["mas_name"] == "MetaGPT" for i in items)


def test_item_ids_are_unique(full_path):
    items = load_mast_items(full_path, target="any")
    ids = [i.item_id for i in items]
    assert len(ids) == len(set(ids))


def test_describe_counts(full_path):
    info = describe(full_path, target="any")
    assert info["n_records"] == 30
    assert info["n_error"] == 18
    assert info["n_clean"] == 12
    assert info["per_mode_counts"]["2.4"] == 10
    assert info["per_mode_counts"]["3.2"] == 8
    assert info["total_mode_flags"] == 18
    assert abs(info["per_mode_prevalence"]["2.4"] - 10 / 30) < 1e-4
    assert abs(info["per_mode_share_of_flags"]["2.4"] - 10 / 18) < 1e-4


def test_schema_change_is_reported_not_guessed(tmp_path):
    bad = tmp_path / "MAD_full_dataset.json"
    bad.write_text(json.dumps([{"mas_name": "X", "trace": "y"}]), encoding="utf-8")
    with pytest.raises(MastSchemaError):
        describe(str(bad))


def test_label_noise_report(human_path):
    report = analyse_label_noise(human_path)
    assert report.n_traces == 5
    assert report.n_annotations == 15
    assert report.per_mode["1.3"]["unanimous_rate"] == 1.0
    assert report.per_mode["2.4"]["unanimous_rate"] == 0.0
    assert report.per_mode["2.4"]["majority_positive_rate"] == 1.0
    assert report.per_mode["3.2"]["unanimous_rate"] == 1.0
    assert abs(report.overall_full_agreement - 2 / 3) < 1e-9


def test_label_noise_separates_trivial_from_informative(human_path):
    report = analyse_label_noise(human_path)
    assert report.per_mode["3.2"]["n_informative"] == 0
    assert report.per_mode["1.3"]["n_informative"] == 5
    assert report.per_mode["1.3"]["informative_unanimous_rate"] == 1.0
    assert report.per_mode["2.4"]["informative_unanimous_rate"] == 0.0
    assert report.n_informative == 10
    assert abs(report.overall_full_agreement_informative - 0.5) < 1e-9


def test_label_noise_flags_taxonomy_drift(tmp_path):
    records = [{
        "round": "Round 1", "mas_name": "X", "benchmark_name": "Y", "trace_id": 1,
        "trace": "t",
        "annotations": [
            {"annotator_1": True, "annotator_2": True, "annotator_3": True,
             "failure mode": "1.1 Disobey Task Specification\nsome definition"},
            {"annotator_1": False, "annotator_2": False, "annotator_3": False,
             "failure mode": "1.1 Poor task constraint compliance\nother definition"},
            {"annotator_1": True, "annotator_2": False, "annotator_3": False,
             "failure mode": "4.3 Lack of critical verification"},
        ],
    }]
    path = tmp_path / "MAD_human_labelled_dataset.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    report = analyse_label_noise(str(path))
    assert "4.3" in report.ids_outside_final_taxonomy
    assert "1.1" in report.ids_with_multiple_titles
    assert len(report.ids_with_multiple_titles["1.1"]) == 2
    assert report.rounds == {"Round 1": 1}


def test_verifier_prompt_tracks_target():
    any_p = system_prompt_for("any")
    fc3_p = system_prompt_for("fc3")
    mode_p = system_prompt_for("2.4")
    assert any_p != fc3_p != mode_p
    assert "for any reason" in any_p
    assert "verification" in fc3_p
    assert "hold back information" in mode_p
    for p in (any_p, fc3_p, mode_p):
        assert "VERDICT: FAILURE" in p and "VERDICT: NO_FAILURE" in p


def test_verifier_prompt_rejects_unknown_target():
    with pytest.raises(ValueError):
        system_prompt_for("9.9")
