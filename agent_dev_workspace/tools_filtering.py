from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning tools for feature filtering operations."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Filters feature matrix terms based on variance threshold.

        Computes population variance (ddof=0) for each term across all documents in session.feature_matrix.
        Keeps terms with variance strictly greater than threshold (> threshold).

        Args:
            threshold: Minimum variance cutoff. Terms with variance > threshold are kept. Defaults to 0.0.

        Returns:
            Dict containing summary counts, previews of kept and removed terms, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first."}

        # Convert sparse matrix to dense array or compute variance across rows (axis=0)
        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        # Compute population variance (ddof=0) matching numpy/pandas default population variance on fixture
        variances = np.var(dense_matrix, axis=0, ddof=0)

        df = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        })

        # Sort descending by variance as required by visualize_result_tool contract
        df_sorted = df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Filter: keep terms with variance > threshold
        kept_df = df_sorted[df_sorted["variance"] > threshold]
        removed_df = df_sorted[df_sorted["variance"] <= threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        res_id = session.next_result_id("variance_filter")
        args = {"threshold": threshold}

        summary = {
            "result_id": res_id,
            "threshold": threshold,
            "n_total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms_preview": kept_terms[:10],
            "removed_terms_preview": removed_terms[:10]
        }

        # Store full report DataFrame for visualize_result_tool (1st col: term, 2nd col: variance, sorted)
        session.store_result(
            "variance_filter_tool",
            args=args,
            summary=summary,
            full_report=df_sorted
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculates Pearson correlation between each term's frequency and a one-vs-rest binary target class.

        Sorts terms by absolute Pearson correlation (|pearson_r|) descending.
        Keeps terms where |pearson_r| > threshold.

        Args:
            target_class: The category name to correlate against (binary 1 for target_class, 0 for other categories).
            threshold: Correlation magnitude cutoff. Terms with |pearson_r| > threshold are kept. Defaults to 0.0.

        Returns:
            Dict containing summary counts, previews of kept and removed terms, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No category_name column in session.dataframe to construct target class vector."}

        # Binary target vector
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        pearson_rs = []
        y_std = np.std(y)

        for col_idx in range(dense_matrix.shape[1]):
            x = dense_matrix[:, col_idx]
            x_std = np.std(x)
            if x_std == 0 or y_std == 0:
                pearson_rs.append(0.0)
            else:
                r = np.corrcoef(x, y)[0, 1]
                pearson_rs.append(0.0 if np.isnan(r) else float(r))

        df = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": pearson_rs,
            "abs_pearson_r": np.abs(pearson_rs)
        })

        # Sort descending by absolute pearson_r so top correlations (positive or negative) come first
        df_sorted = df.sort_values(by="abs_pearson_r", ascending=False).reset_index(drop=True)

        # Filter strictly > threshold on magnitude
        kept_df = df_sorted[df_sorted["abs_pearson_r"] > threshold]
        removed_df = df_sorted[df_sorted["abs_pearson_r"] <= threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        res_id = session.next_result_id("pearson_filter")
        args = {"target_class": target_class, "threshold": threshold}

        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms_preview": kept_terms[:10],
            "removed_terms_preview": removed_terms[:10]
        }

        # Prepare full_report DataFrame for visualize_result_tool (1st col: term, 2nd col: pearson_r)
        report_df = df_sorted[["term", "pearson_r"]]

        session.store_result(
            "pearson_filter_tool",
            args=args,
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        max_terms: int = 1500
    ) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between each term's frequency and a one-vs-rest binary target class.

        Optionally narrows down terms to top max_terms by variance first for performance.
        Sorts terms by absolute Spearman correlation (|spearman_r|) descending.
        Keeps terms where |spearman_r| > threshold.

        Args:
            target_class: The category name to correlate against (binary 1 for target_class, 0 for other categories).
            threshold: Correlation magnitude cutoff. Terms with |spearman_r| > threshold are kept. Defaults to 0.0.
            max_terms: Maximum number of top-variance terms to consider before calculating rank correlation. Defaults to 1500.

        Returns:
            Dict containing summary counts, previews of kept and removed terms, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first."}
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No category_name column in session.dataframe to construct target class vector."}

        # Binary target vector
        y = (session.dataframe["category_name"] == target_class).astype(float).values

        matrix = session.feature_matrix
        if hasattr(matrix, "toarray"):
            dense_matrix = matrix.toarray()
        else:
            dense_matrix = np.asarray(matrix)

        total_terms = dense_matrix.shape[1]

        # Top-variance pre-filtering if total_terms exceeds max_terms
        if max_terms > 0 and total_terms > max_terms:
            variances = np.var(dense_matrix, axis=0, ddof=0)
            top_indices = np.argsort(variances)[::-1][:max_terms]
        else:
            top_indices = np.arange(total_terms)

        spearman_rs = []
        selected_terms = [session.feature_names[i] for i in top_indices]
        y_std = np.std(y)

        for col_idx in top_indices:
            x = dense_matrix[:, col_idx]
            x_std = np.std(x)
            if x_std == 0 or y_std == 0:
                spearman_rs.append(0.0)
            else:
                res = spearmanr(x, y)
                corr = res.statistic if hasattr(res, "statistic") else res[0]
                spearman_rs.append(0.0 if np.isnan(corr) else float(corr))

        df = pd.DataFrame({
            "term": selected_terms,
            "spearman_r": spearman_rs,
            "abs_spearman_r": np.abs(spearman_rs)
        })

        # Sort descending by absolute spearman_r so top correlations (positive or negative) come first
        df_sorted = df.sort_values(by="abs_spearman_r", ascending=False).reset_index(drop=True)

        # Filter strictly > threshold on magnitude
        kept_df = df_sorted[df_sorted["abs_spearman_r"] > threshold]
        removed_df = df_sorted[df_sorted["abs_spearman_r"] <= threshold]

        kept_terms = kept_df["term"].tolist()
        removed_terms = removed_df["term"].tolist()

        res_id = session.next_result_id("spearman_filter")
        args = {"target_class": target_class, "threshold": threshold, "max_terms": max_terms}

        summary = {
            "result_id": res_id,
            "target_class": target_class,
            "threshold": threshold,
            "max_terms": max_terms,
            "n_total_terms": total_terms,
            "n_evaluated": len(selected_terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms_preview": kept_terms[:10],
            "removed_terms_preview": removed_terms[:10]
        }

        # Prepare full_report DataFrame for visualize_result_tool (1st col: term, 2nd col: spearman_r)
        report_df = df_sorted[["term", "spearman_r"]]

        session.store_result(
            "spearman_filter_tool",
            args=args,
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
