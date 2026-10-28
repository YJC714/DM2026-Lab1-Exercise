from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Aggregates total term frequencies across all documents in the document-term matrix.

        Args:
            top_n: Optional number of top terms to include in the returned term frequencies list.
                   Defaults to 10 if not provided or None.

        Returns:
            Dict containing result_id, total_terms, and term_frequencies (list of dicts with 'term' and 'frequency').
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Please run build_dtm_tool first."}

        # Sum along columns (axis=0) to get total occurrence count per term
        term_counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        
        # Build list of (term, frequency) dicts
        freq_list = [
            {"term": term, "frequency": int(count)}
            for term, count in zip(session.feature_names, term_counts)
        ]
        
        # Sort descending by frequency
        freq_list.sort(key=lambda x: x["frequency"], reverse=True)

        # Default limit to 10 terms in summary if top_n is not explicitly set
        limit = top_n if (top_n is not None and top_n > 0) else 10
        reported_freqs = freq_list[:limit]

        res_id = session.next_result_id("term_freq")
        args = {"top_n": top_n}

        summary = {
            "result_id": res_id,
            "total_terms": len(session.feature_names),
            "term_frequencies": reported_freqs
        }

        # Full report contains all terms in sorted order for full-data downstream tasks
        df_report = pd.DataFrame(freq_list)
        session.store_result("term_frequency_tool", args, summary, full_report=df_report)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generates a raw positional heatmap slice of the document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include in the heatmap. Defaults to 20.
            n_documents: Number of documents (rows) to include in the heatmap. Defaults to 20.

        Returns:
            Dict containing result_id, n_documents, n_terms, term_labels, doc_labels, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Please run build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape
        actual_n_docs = min(n_documents, total_docs)
        actual_n_terms = min(n_terms, total_terms)

        # Slice submatrix and convert to dense array
        dense_matrix = session.feature_matrix[:actual_n_docs, :actual_n_terms].toarray()
        term_labels = session.feature_names[:actual_n_terms]

        # Construct doc_labels including category_name if available
        doc_labels = []
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            cat_names = session.dataframe["category_name"].iloc[:actual_n_docs].tolist()
            for i in range(actual_n_docs):
                cat = cat_names[i] if i < len(cat_names) else ""
                doc_labels.append(f"Doc {i} ({cat})")
        else:
            for i in range(actual_n_docs):
                doc_labels.append(f"Doc {i}")

        # Build figure and heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.6), max(4, actual_n_docs * 0.4)))
        sns.heatmap(
            dense_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=term_labels,
            yticklabels=doc_labels,
            ax=ax
        )
        ax.set_title(f"Document-Term Matrix Heatmap ({actual_n_docs} Docs x {actual_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure for the chat/pipeline UI
        session.pending_figure = fig
        plt.close(fig)

        res_id = session.next_result_id("dtm_heatmap")
        args = {"n_terms": n_terms, "n_documents": n_documents}

        matrix_list = dense_matrix.tolist()

        summary = {
            "result_id": res_id,
            "n_documents": actual_n_docs,
            "n_terms": actual_n_terms,
            "term_labels": term_labels,
            "doc_labels": doc_labels,
            "matrix_slice": matrix_list
        }

        session.store_result("dtm_heatmap_tool", args, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
