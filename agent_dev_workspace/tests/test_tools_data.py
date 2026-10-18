import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools
from tools_dtm import make_tools as make_dtm_tools


def test_check_missing_tool_known_answer_2():
    session = SessionState()

    # Load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # Instantiate student's check_missing_tool from tools_data.py
    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    # Invoke check_missing_tool
    res = check_missing_tool.invoke({"column_name": "text"})

    # Assert against known answer #2: exactly 1 row missing/empty at index 7
    assert res["n_total_documents"] == 8
    assert res["n_missing"] == 1
    assert res["missing_indices_preview"] == [7]

    # Check that session.dataframe is untouched and still has 8 rows
    assert session.dataframe is not None
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_known_answer_3():
    session = SessionState()

    # Load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    # 1. Test inspection mode (drop=False)
    res_inspect = check_duplicates_tool.invoke({"column_name": "text", "drop": False})

    assert res_inspect["n_total_before"] == 8
    assert res_inspect["n_duplicates"] == 1
    assert res_inspect["duplicate_indices_preview"] == [6]
    assert res_inspect["n_total_after"] == 8
    assert len(session.dataframe) == 8

    # Build DTM matrix to check that dropping clears feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    assert session.feature_matrix is not None

    # 2. Test dropping mode (drop=True)
    res_drop = check_duplicates_tool.invoke({"column_name": "text", "drop": True})

    # Assert against known answer #3: keep=False drops both row 5 and row 6, leaving 6 rows
    assert res_drop["n_total_before"] == 8
    assert res_drop["n_duplicates"] == 1
    assert res_drop["n_total_after"] == 6
    assert len(session.dataframe) == 6

    # Verify that feature_matrix was cleared and note was provided
    assert session.feature_matrix is None
    assert session.feature_names is None
    assert "note" in res_drop
    assert "rebuild DTM" in res_drop["note"]


def test_describe_data_tool_known_answer_18():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({
        "text_column": "text",
        "category_column": "category_name"
    })

    # 1. Assert box plot was created on session.pending_figure
    assert session.pending_figure is not None

    # 2. Assert session.dataframe remains untouched (8 rows, original columns)
    assert len(session.dataframe) == 8
    assert "text_length" not in session.dataframe.columns

    # 3. Assert overall stats against known answer #18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330
    assert overall["min"] == 0
    assert pytest.approx(overall["25%"], abs=1e-3) == 17.75
    assert pytest.approx(overall["50%"], abs=1e-3) == 20.5
    assert pytest.approx(overall["75%"], abs=1e-3) == 25.25
    assert overall["max"] == 30

    # 4. Assert per-category stats against known answer #18
    per_cat = res["per_category_stats"]
    assert "catA" in per_cat
    assert "catB" in per_cat

    catA = per_cat["catA"]
    assert catA["count"] == 4
    assert pytest.approx(catA["mean"], abs=1e-3) == 17.25
    assert pytest.approx(catA["std"], abs=1e-3) == 12.5
    assert catA["min"] == 0
    assert pytest.approx(catA["25%"], abs=1e-3) == 12.75
    assert pytest.approx(catA["50%"], abs=1e-3) == 20.0
    assert pytest.approx(catA["75%"], abs=1e-3) == 24.5
    assert catA["max"] == 29

    catB = per_cat["catB"]
    assert catB["count"] == 4
    assert pytest.approx(catB["mean"], abs=1e-3) == 22.5
    assert pytest.approx(catB["std"], abs=1e-3) == 5.7446
    assert catB["min"] == 18
    assert pytest.approx(catB["25%"], abs=1e-3) == 18.0
    assert pytest.approx(catB["50%"], abs=1e-3) == 21.0
    assert pytest.approx(catB["75%"], abs=1e-3) == 25.5
    assert catB["max"] == 30
