from typing import Dict, Any
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning tools for dataset inspection and data quality operations."""

    @tool
    def check_missing_tool(column_name: str = "text") -> Dict[str, Any]:
        """Checks for missing or empty text values in session.dataframe.

        Identifies both null/NaN values and empty or whitespace-only strings without modifying session.dataframe.

        Args:
            column_name: The DataFrame column to inspect for missing/empty values. Defaults to "text".

        Returns:
            Dict containing total document count, missing count, a preview of missing row indices, and result_id.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session state."}
        if column_name not in session.dataframe.columns:
            return {"error": f"Column '{column_name}' not found in session.dataframe."}

        df = session.dataframe

        # Detect null/NaN or empty/whitespace-only strings
        is_null = df[column_name].isnull()
        is_empty_str = df[column_name].astype(str).str.strip() == ""
        missing_mask = is_null | is_empty_str

        missing_indices = df.index[missing_mask].tolist()
        n_missing = len(missing_indices)
        n_total = len(df)

        res_id = session.next_result_id("check_missing")
        args = {"column_name": column_name}

        summary = {
            "result_id": res_id,
            "column_name": column_name,
            "n_total_documents": n_total,
            "n_missing": n_missing,
            "missing_indices_preview": missing_indices[:10]
        }

        session.store_result("check_missing_tool", args=args, summary=summary)

        return summary

    @tool
    def check_duplicates_tool(
        column_name: str = "text",
        drop: bool = False
    ) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe and optionally drops them.

        When drop=True, removes duplicate documents using keep=False (dropping all copies of duplicate rows),
        clears session.feature_matrix and session.feature_names to prevent state mismatch,
        and prompts the user/agent to rebuild the DTM.

        Args:
            column_name: The DataFrame column to check for duplicates. Defaults to "text".
            drop: If True, drops duplicate rows in place using keep=False. Defaults to False.

        Returns:
            Dict containing initial row count, duplicate count, final row count, duplicate index preview, and result_id.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session state."}
        if column_name not in session.dataframe.columns:
            return {"error": f"Column '{column_name}' not found in session.dataframe."}

        df = session.dataframe
        n_total_before = len(df)

        # Count extra duplicate rows using keep='first'
        dup_mask = df.duplicated(subset=[column_name], keep="first")
        duplicate_indices = df.index[dup_mask].tolist()
        n_duplicates = len(duplicate_indices)

        note = None

        if drop and n_duplicates > 0:
            # Drop all copies of duplicated rows matching spec (keep=False)
            df.drop_duplicates(subset=[column_name], keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Clear outdated DTM matrix state
            session.feature_matrix = None
            session.feature_names = None

            if "category_name" in df.columns:
                session.set_labels(df["category_name"].to_numpy())

            note = (
                "Dropped duplicate documents. Cleared session.feature_matrix and session.feature_names. "
                "Please rebuild DTM with build_dtm_tool before running downstream filtering/exploration tools."
            )

        n_total_after = len(session.dataframe)

        res_id = session.next_result_id("check_duplicates")
        args = {"column_name": column_name, "drop": drop}

        summary = {
            "result_id": res_id,
            "column_name": column_name,
            "drop": drop,
            "n_total_before": n_total_before,
            "n_duplicates": n_duplicates,
            "n_total_after": n_total_after,
            "duplicate_indices_preview": duplicate_indices[:10]
        }

        if note:
            summary["note"] = note

        session.store_result("check_duplicates_tool", args=args, summary=summary)

        return summary

    return [check_missing_tool, check_duplicates_tool]
