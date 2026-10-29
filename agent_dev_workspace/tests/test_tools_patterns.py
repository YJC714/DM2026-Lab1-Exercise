import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_known_answers_17_and_20():
    session = SessionState()

    # Load fixture data
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # 1. Validation check: fpgrowth without min_sup should return error
    err_res = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})
    assert "error" in err_res
    assert "min_sup parameter is required" in err_res["error"]

    # 2. Validation check: topk without k should return error
    err_res_k = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})
    assert "error" in err_res_k
    assert "k parameter is required" in err_res_k["error"]

    # 3. Known Answer #17: Zero terms survive for variance and term_frequency filtering
    for filter_m in ["variance", "term_frequency"]:
        res_zero_a = mine_patterns_tool.invoke({
            "category_name": "catA",
            "filtering_method": filter_m,
            "algorithm": "fpgrowth",
            "min_sup": 1
        })
        assert res_zero_a["n_patterns"] == 0 if "n_patterns" in res_zero_a else len(res_zero_a["patterns"]) == 0
        assert "no terms survived" in res_zero_a.get("note", "")

        res_zero_b = mine_patterns_tool.invoke({
            "category_name": "catB",
            "filtering_method": filter_m,
            "algorithm": "fpgrowth",
            "min_sup": 1
        })
        assert res_zero_b["n_patterns"] == 0 if "n_patterns" in res_zero_b else len(res_zero_b["patterns"]) == 0
        assert "no terms survived" in res_zero_b.get("note", "")

    # 4. Known Answer #17: TF-IDF + FPGrowth
    # catA -> alpha (support 3)
    res_fpgrowth_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_fpgrowth_cata["n_patterns"] == 1
    p_cata = res_fpgrowth_cata["patterns_preview"][0]
    assert p_cata["pattern"] == ["alpha"]
    assert p_cata["support"] == 3

    # catB -> gamma (support 4)
    res_fpgrowth_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_fpgrowth_catb["n_patterns"] == 1
    p_catb = res_fpgrowth_catb["patterns_preview"][0]
    assert p_catb["pattern"] == ["gamma"]
    assert p_catb["support"] == 4

    # Check artifact storage
    assert "patterns_catA_tfidf_fpgrowth" in session.artifacts
    assert "patterns_catB_tfidf_fpgrowth" in session.artifacts

    # 5. Known Answer #20: TF-IDF + Top-K (FAE)
    # Both categories return 1 pattern for k=1, k=2, or k=3
    res_topk_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_cata["n_patterns"] == 1
    assert res_topk_cata["patterns_preview"][0]["pattern"] == ["alpha"]
    assert res_topk_cata["patterns_preview"][0]["support"] == 3

    res_topk_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 2
    })
    assert res_topk_catb["n_patterns"] == 1
    assert res_topk_catb["patterns_preview"][0]["pattern"] == ["gamma"]
    assert res_topk_catb["patterns_preview"][0]["support"] == 4

    # 6. Known Answer #20: TF-IDF + MaxFPGrowth
    # MaxFPGrowth returns 0 patterns for single-surviving-term per category (expected PAMI behavior)
    res_max_cata = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_cata["n_patterns"] == 0

    res_max_catb = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catb["n_patterns"] == 0


def test_mine_patterns_tool_unicode_accent_handling():
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["café latte", "café mocha", "café espresso", "café americano"],
        "category_name": ["coffee", "coffee", "coffee", "coffee"]
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "coffee",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert "error" not in res
    assert res["n_patterns"] > 0
    patterns = [p["pattern"] for p in res["patterns_preview"]]
    assert any("cafe" in pattern for pattern in patterns)
