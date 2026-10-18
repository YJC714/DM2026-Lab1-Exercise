from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning tools for document-term matrix operations."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: float = 1.0,
        min_df: int = 1,
        max_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from document text in session.dataframe using CountVectorizer.

        Updates session.feature_matrix, session.feature_names, and session.artifacts['count_vectorizer'].
        Calculates matrix sparsity metrics.

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams to be extracted. Defaults to [1, 1].
            max_df: Ignore terms that have a document frequency strictly higher than the given threshold. Defaults to 1.0.
            min_df: Ignore terms that have a document frequency strictly lower than the given threshold. Defaults to 1.
            max_features: If not None, build a vocabulary that only consider the top max_features ordered by term frequency. Defaults to None.

        Returns:
            Dict containing summary statistics of the built matrix, including matrix shape, sparsity, and a preview of feature names.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session state."}
        if "text" not in session.dataframe.columns:
            return {"error": "DataFrame does not contain a 'text' column."}

        vectorizer = CountVectorizer(
            ngram_range=tuple(ngram_range),
            max_df=max_df,
            min_df=min_df,
            max_features=max_features
        )

        # Fill missing/NaN text with empty string to avoid fit errors
        text_series = session.dataframe["text"].fillna("")
        feature_matrix = vectorizer.fit_transform(text_series)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        res_id = session.next_result_id("dtm")
        args = {
            "ngram_range": ngram_range,
            "max_df": max_df,
            "min_df": min_df,
            "max_features": max_features
        }

        summary = {
            "result_id": res_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names_preview": feature_names[:10]
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    return [build_dtm_tool]
