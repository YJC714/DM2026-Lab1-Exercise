from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning tools for exploratory analysis."""

    @tool
    def cosine_similarity_tool(
        doc_idx1: Optional[int] = None,
        doc_idx2: Optional[int] = None,
        doc_text1: Optional[str] = None,
        doc_text2: Optional[str] = None
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents (by matrix row index or raw text).

        Uses session.artifacts['count_vectorizer'] to vectorize raw text input if provided.
        Otherwise extracts document rows from session.feature_matrix.

        Args:
            doc_idx1: 0-based index of the first document in session.feature_matrix.
            doc_idx2: 0-based index of the second document in session.feature_matrix.
            doc_text1: Raw text string for document 1 (used if doc_idx1 is not provided).
            doc_text2: Raw text string for document 2 (used if doc_idx2 is not provided).

        Returns:
            Dict containing result_id, doc1_info, doc2_info, and cosine_similarity.
        """
        if session.feature_matrix is None and (doc_text1 is None or doc_text2 is None):
            return {"error": "No feature matrix in session state. Run build_dtm_tool first."}

        vectorizer = session.artifacts.get("count_vectorizer")

        # Resolve vector 1
        if doc_idx1 is not None:
            if session.feature_matrix is None:
                return {"error": "session.feature_matrix is missing for doc_idx1."}
            if doc_idx1 < 0 or doc_idx1 >= session.feature_matrix.shape[0]:
                return {"error": f"doc_idx1 out of bounds: {doc_idx1}"}
            vec1 = session.feature_matrix[doc_idx1]
            if session.dataframe is not None and "category_name" in session.dataframe.columns and doc_idx1 < len(session.dataframe):
                cat1 = session.dataframe["category_name"].iloc[doc_idx1]
                doc1_info = f"Doc {doc_idx1} ({cat1})"
            else:
                doc1_info = f"Doc {doc_idx1}"
        elif doc_text1 is not None:
            if vectorizer is None:
                return {"error": "count_vectorizer missing in session.artifacts. Run build_dtm_tool first."}
            vec1 = vectorizer.transform([doc_text1])
            doc1_info = f"Text: '{doc_text1[:30]}...'" if len(doc_text1) > 30 else f"Text: '{doc_text1}'"
        else:
            return {"error": "Must provide either doc_idx1 or doc_text1."}

        # Resolve vector 2
        if doc_idx2 is not None:
            if session.feature_matrix is None:
                return {"error": "session.feature_matrix is missing for doc_idx2."}
            if doc_idx2 < 0 or doc_idx2 >= session.feature_matrix.shape[0]:
                return {"error": f"doc_idx2 out of bounds: {doc_idx2}"}
            vec2 = session.feature_matrix[doc_idx2]
            if session.dataframe is not None and "category_name" in session.dataframe.columns and doc_idx2 < len(session.dataframe):
                cat2 = session.dataframe["category_name"].iloc[doc_idx2]
                doc2_info = f"Doc {doc_idx2} ({cat2})"
            else:
                doc2_info = f"Doc {doc_idx2}"
        elif doc_text2 is not None:
            if vectorizer is None:
                return {"error": "count_vectorizer missing in session.artifacts. Run build_dtm_tool first."}
            vec2 = vectorizer.transform([doc_text2])
            doc2_info = f"Text: '{doc_text2[:30]}...'" if len(doc_text2) > 30 else f"Text: '{doc_text2}'"
        else:
            return {"error": "Must provide either doc_idx2 or doc_text2."}

        # Compute cosine similarity
        sim = cosine_similarity(vec1, vec2)[0][0]
        # Handle NaN output when vector norm is zero (empty text)
        if np.isnan(sim):
            sim_score = 0.0
        else:
            sim_score = float(sim)

        res_id = session.next_result_id("cosine_sim")
        args = {
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "doc_text1": doc_text1,
            "doc_text2": doc_text2
        }

        summary = {
            "result_id": res_id,
            "doc1_info": doc1_info,
            "doc2_info": doc2_info,
            "cosine_similarity": round(sim_score, 4)
        }

        session.store_result("cosine_similarity_tool", args, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Computes Pearson feature-vs-feature correlation matrix for top terms by variance.

        Generates a correlation heatmap and assigns it to session.pending_figure.

        Args:
            top_n: Number of highest variance terms to include in correlation matrix. Defaults to 20.

        Returns:
            Dict containing result_id, n_terms, selected_terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Run build_dtm_tool first."}

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        # Compute term variance with ddof=0 matching variance_filter_tool
        variances = np.var(dense_matrix, axis=0, ddof=0)
        
        # Sort terms by variance descending
        sorted_indices = np.argsort(variances)[::-1]
        
        # Take top_n indices
        actual_top_n = min(top_n, len(session.feature_names))
        selected_indices = sorted_indices[:actual_top_n]

        selected_terms = [session.feature_names[i] for i in selected_indices]
        selected_submatrix = dense_matrix[:, selected_indices]

        # Compute Pearson correlation matrix across selected term columns
        corr_matrix = np.corrcoef(selected_submatrix, rowvar=False)

        # Handle scalar (single term) output from np.corrcoef
        if np.ndim(corr_matrix) == 0:
            corr_matrix = np.array([[1.0]])

        # Handle zero-variance terms resulting in NaNs
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        # Plot annotated heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_top_n * 0.8), max(5, actual_top_n * 0.7)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        res_id = session.next_result_id("feat_corr")
        args = {"top_n": top_n}

        corr_list = np.round(corr_matrix, 4).tolist()

        summary = {
            "result_id": res_id,
            "n_terms": actual_top_n,
            "selected_terms": selected_terms,
            "correlation_matrix": corr_list
        }

        df_report = pd.DataFrame(corr_matrix, index=selected_terms, columns=selected_terms)
        session.store_result("feature_correlation_matrix_tool", args, summary, full_report=df_report)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
