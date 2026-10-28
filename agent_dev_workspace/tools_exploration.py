from typing import Dict, Any, Optional
import numpy as np
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

    return [cosine_similarity_tool]
