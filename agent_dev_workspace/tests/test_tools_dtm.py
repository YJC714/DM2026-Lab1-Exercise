import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    # 1. Instantiate single session state
    session = SessionState()

    # 2. Load premade load_dataset_tool to load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # 3. Instantiate student's tool from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # 4. Invoke build_dtm_tool
    res = build_dtm_tool.invoke({})

    # 5. Assert summary metrics match known answer #4
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert pytest.approx(res["sparsity_pct"], abs=1e-3) == 47.5

    # 6. Assert full feature names on session match known answer #4
    expected_terms = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_terms

    # 7. Assert vectorizer artifact was stored
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    # 1. Instantiate fresh session state
    session = SessionState()

    # 2. Load premade load_dataset_tool to load sample_fixture.csv
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student's tools from tools_dtm
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # 4. Build DTM first
    build_dtm_tool.invoke({})

    # 5. Invoke term_frequency_tool with default parameters
    res = term_frequency_tool.invoke({})
    assert res["total_terms"] == 5

    # Verify counts match known answer #14
    freq_dict = {item["term"]: item["frequency"] for item in res["term_frequencies"]}
    expected_counts = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    assert freq_dict == expected_counts

    # Check 1: Verify full_report columns and descending sort
    result_id = res["result_id"]
    stored_entry = session.results_store[result_id]
    full_report = stored_entry["full_report"]

    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "frequency"]
    frequencies = full_report["frequency"].tolist()
    assert frequencies == sorted(frequencies, reverse=True)

    # Check 2: Test top_n=2 limiting in summary
    res_top2 = term_frequency_tool.invoke({"top_n": 2})
    top2_terms = [item["term"] for item in res_top2["term_frequencies"]]
    assert len(top2_terms) == 2
    assert set(top2_terms) == {"always", "gamma"}


def test_dtm_heatmap_tool_known_answer_21():
    # 1. Instantiate fresh session state
    session = SessionState()

    # 2. Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # 4. Build DTM
    build_dtm_tool.invoke({})

    # 5. Invoke dtm_heatmap_tool with defaults (n_terms=20, n_documents=20)
    res = dtm_heatmap_tool.invoke({})

    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["term_labels"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Verify doc labels contain category names
    assert res["doc_labels"][0] == "Doc 0 (catA)"
    assert res["doc_labels"][3] == "Doc 3 (catB)"

    # Verify full matrix slice matches Known Answer #21
    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res["matrix_slice"] == expected_matrix

    # Verify plot figure was created
    assert session.pending_figure is not None

    # 6. Test slicing with n_terms=2, n_documents=3
    res_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert res_sliced["n_documents"] == 3
    assert res_sliced["n_terms"] == 2
    assert res_sliced["term_labels"] == ["alpha", "always"]
    assert res_sliced["matrix_slice"] == [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
