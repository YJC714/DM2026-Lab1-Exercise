import pytest
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
