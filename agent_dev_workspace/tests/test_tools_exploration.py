import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_known_answer_16():
    session = SessionState()

    # Load dataset and build DTM
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # 0 vs 1 (both catA, near-identical text)
    res01 = cosine_sim_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert res01["doc1_info"] == "Doc 0 (catA)"
    assert res01["doc2_info"] == "Doc 1 (catA)"
    assert pytest.approx(res01["cosine_similarity"], abs=1e-3) == 0.9847

    # 0 vs 3 (catA vs catB, share only "always")
    res03 = cosine_sim_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res03["doc2_info"] == "Doc 3 (catB)"
    assert pytest.approx(res03["cosine_similarity"], abs=1e-3) == 0.0909

    # 3 vs 4 (both catB, near-identical text)
    res34 = cosine_sim_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert pytest.approx(res34["cosine_similarity"], abs=1e-3) == 0.9847

    # 0 vs 7 (catA vs empty text row)
    res07 = cosine_sim_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res07["cosine_similarity"] == 0.0


def test_cosine_similarity_raw_text():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Compare Doc 0 text vs Doc 1 text explicitly (should match 0 vs 1 index result: 0.9847)
    doc_text0 = "always alpha alpha alpha beta"
    doc_text1 = "always alpha alpha beta"

    res_texts = cosine_sim_tool.invoke({"doc_text1": doc_text0, "doc_text2": doc_text1})
    assert pytest.approx(res_texts["cosine_similarity"], abs=1e-3) == 0.9847

    # Mixed comparison: doc_idx1 = 0 and doc_text2 = doc_text1
    res_mixed = cosine_sim_tool.invoke({"doc_idx1": 0, "doc_text2": doc_text1})
    assert pytest.approx(res_mixed["cosine_similarity"], abs=1e-3) == 0.9847


def test_cosine_similarity_out_of_bounds():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Test out-of-bounds index
    res_err = cosine_sim_tool.invoke({"doc_idx1": 100, "doc_idx2": 0})
    assert "error" in res_err
