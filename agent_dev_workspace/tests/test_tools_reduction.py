import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_pca_known_answer_7():
    session = SessionState()

    # Load fixture data and build DTM
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Check pending_figure plot requirement
    assert session.pending_figure is not None

    # Check artifacts storage
    assert "reduced_pca" in session.artifacts
    coords = session.artifacts["reduced_pca"]
    assert coords.shape == (8, 2)

    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    np.testing.assert_allclose(coords, expected_pca, atol=1e-3)

    # Check explained_variance_ratio
    expected_exp_var = [0.7532, 0.1965]
    np.testing.assert_allclose(res["explained_variance_ratio"], expected_exp_var, atol=1e-3)


# Note: We do not assert against exact coordinate values for known answer #8 (t-SNE) and #9 (UMAP)
# because non-deterministic manifold learning algorithms vary across library versions and BLAS/LAPACK backends.
# Instead, we verify shape, plot generation on session.pending_figure, artifact storage,
# and deterministic reproducibility when random_state is fixed.

def test_tsne_reproducibility_and_plot():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Call twice with same random_state=42 to verify reproducibility and plot generation
    res1 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })
    coords1 = session.artifacts["reduced_tsne"].copy()

    assert session.pending_figure is not None
    assert coords1.shape == (8, 2)

    res2 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })
    coords2 = session.artifacts["reduced_tsne"]

    # Assert deterministic reproducibility when random_state is supplied
    np.testing.assert_allclose(coords1, coords2, atol=1e-5)


def test_umap_reproducibility_and_plot():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Call twice with same random_state=42 to verify reproducibility and plot generation
    res1 = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    coords1 = session.artifacts["reduced_umap"].copy()

    assert session.pending_figure is not None
    assert coords1.shape == (8, 2)

    res2 = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })
    coords2 = session.artifacts["reduced_umap"]

    # Assert deterministic reproducibility when random_state is supplied
    np.testing.assert_allclose(coords1, coords2, atol=1e-5)
