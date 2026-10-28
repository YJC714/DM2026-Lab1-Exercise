from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning tools for dimensionality reduction and label operations."""

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        init: str = "random"
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions using PCA, t-SNE, or UMAP and produces a scatter plot.

        Always sets a scatter plot with category legend on session.pending_figure.
        Stores full reduced coordinates in session.artifacts[f"reduced_{method}"].

        Args:
            method: Reduction algorithm ("pca", "tsne", or "umap"). Defaults to "pca".
            n_components: Target number of dimensions (2 or 3). Defaults to 2.
            random_state: Random seed for reproducible reduction. Defaults to None.
            perplexity: Perplexity parameter for t-SNE. Defaults to 30.0.
            n_neighbors: Number of neighbors parameter for UMAP. Defaults to 15.
            init: Initialization method for t-SNE / UMAP ("random" or "pca"/"spectral"). Defaults to "random".

        Returns:
            Dict containing summary metadata, explained_variance_ratio (PCA), and a preview of 50 coordinates.
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix in session state. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        method_lower = method.lower()
        explained_variance_ratio = None

        if method_lower == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init=init
            )
            coords = model.fit_transform(dense_matrix)
        elif method_lower == "umap":
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
                init=init
            )
            coords = model.fit_transform(dense_matrix)
        else:
            return {"error": f"Unsupported method '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Store full coordinates in session artifacts
        session.artifacts[f"reduced_{method_lower}"] = coords

        # Extract category labels for scatter plot
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].values
        elif session.labels is not None:
            categories = session.labels
        else:
            categories = np.array(["unknown"] * len(coords))

        unique_cats = np.unique(categories)

        # Build Matplotlib figure
        fig = plt.figure(figsize=(8, 6))

        if n_components == 3:
            ax = fig.add_subplot(111, projection="3d")
            for cat in unique_cats:
                mask = categories == cat
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    coords[mask, 2],
                    label=str(cat),
                    alpha=0.7
                )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.set_zlabel("Component 3")
        else:
            ax = fig.add_subplot(111)
            for cat in unique_cats:
                mask = categories == cat
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    label=str(cat),
                    alpha=0.7
                )
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.legend(title="Category")
        plt.tight_layout()

        # Mandatory plotting requirement: set pending_figure
        session.pending_figure = fig

        res_id = session.next_result_id(f"reduce_{method_lower}")
        args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "init": init
        }

        summary = {
            "result_id": res_id,
            "method": method_lower,
            "n_components": n_components,
            "n_documents": len(coords),
            "coordinates_preview": coords.tolist()[:50]
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = [round(v, 4) for v in explained_variance_ratio]

        session.store_result("reduce_dimensions_tool", args=args, summary=summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes category labels into a binary matrix with one column per category.

        Does not modify session.dataframe. Stores full binarized DataFrame in session.artifacts['binarized_labels'].

        Returns:
            Dict containing result_id, n_samples, n_classes, classes, and matrix_preview.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session state."}
        if "category_name" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'category_name' column."}

        category_series = session.dataframe["category_name"]
        
        # One-hot encode using pd.get_dummies, ensuring 1 column per category
        binarized_df = pd.get_dummies(category_series, dtype=int)
        
        classes = list(binarized_df.columns)
        n_samples, n_classes = binarized_df.shape

        # Save to artifacts without modifying session.dataframe
        session.artifacts["binarized_labels"] = binarized_df

        res_id = session.next_result_id("binarize")
        args = {}

        summary = {
            "result_id": res_id,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "classes": classes,
            "matrix_preview": binarized_df.head(10).to_dict(orient="records")
        }

        session.store_result("binarize_labels_tool", args, summary, full_report=binarized_df)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
