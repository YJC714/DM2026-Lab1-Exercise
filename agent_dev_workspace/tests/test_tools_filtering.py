import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # Load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Test with threshold = 0.15 matching known answer #5
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["n_total_terms"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert "always" in res["removed_terms_preview"]
    assert set(res["kept_terms_preview"]) == {"alpha", "beta", "delta", "gamma"}

    # Fetch stored result full_report to verify exact numerical variances and structure from known answer #5
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)

    # 1. Check column order required by visualize_result_tool (1st col: term, 2nd col: variance)
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # 2. Check rows are sorted in descending order by variance
    variances = full_report["variance"].tolist()
    assert variances == sorted(variances, reverse=True)

    # 3. Check exact numerical variances
    variance_map = dict(zip(full_report["term"], full_report["variance"]))
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    for term, exp_var in expected_variances.items():
        assert pytest.approx(variance_map[term], abs=1e-4) == exp_var


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    # Load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # 1. Invoke pearson_filter_tool with threshold=0.0 to check full report and exact r values
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["target_class"] == "catB"
    assert res["n_total_terms"] == 5
    assert res["n_kept"] == 5
    assert res["n_removed"] == 0

    # Fetch full_report from session.results_store
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)

    # Check column order required by visualize_result_tool (1st col: term, 2nd col: pearson_r)
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Check rows are sorted in descending order by |pearson_r|
    abs_rs = full_report["pearson_r"].abs().tolist()
    assert abs_rs == sorted(abs_rs, reverse=True)

    # Verify exact numerical Pearson r values against known answer #6
    pearson_map = dict(zip(full_report["term"], full_report["pearson_r"]))

    expected_pearson_rs = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_r in expected_pearson_rs.items():
        assert pytest.approx(pearson_map[term], abs=1e-3) == exp_r

    # 2. Invoke pearson_filter_tool with threshold=0.5 to test filtering logic
    res_thresh = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert res_thresh["n_total_terms"] == 5
    assert res_thresh["n_kept"] == 4
    assert res_thresh["n_removed"] == 1
    assert set(res_thresh["kept_terms_preview"]) == {"delta", "gamma", "beta", "alpha"}
    assert res_thresh["removed_terms_preview"] == ["always"]


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    # Load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. Invoke spearman_filter_tool with threshold=0.0
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["target_class"] == "catB"
    assert res["n_total_terms"] == 5
    assert res["n_kept"] == 5
    assert res["n_removed"] == 0

    # Fetch full_report from session.results_store
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]
    assert isinstance(full_report, pd.DataFrame)

    # Check column order required by visualize_result_tool (1st col: term, 2nd col: spearman_r)
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Check rows are sorted in descending order by |spearman_r|
    abs_rs = full_report["spearman_r"].abs().tolist()
    assert abs_rs == sorted(abs_rs, reverse=True)

    # Verify exact numerical Spearman r values against known answer #6
    spearman_map = dict(zip(full_report["term"], full_report["spearman_r"]))

    expected_spearman_rs = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, exp_r in expected_spearman_rs.items():
        assert pytest.approx(spearman_map[term], abs=1e-3) == exp_r

    # 2. Invoke spearman_filter_tool with threshold=0.5
    res_thresh = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.5})

    assert res_thresh["n_total_terms"] == 5
    assert res_thresh["n_kept"] == 4
    assert res_thresh["n_removed"] == 1
    assert set(res_thresh["kept_terms_preview"]) == {"delta", "gamma", "beta", "alpha"}
    assert res_thresh["removed_terms_preview"] == ["always"]
