from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
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

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name"
    ) -> Dict[str, Any]:
        """Calculates text character length summary stats and generates a box plot grouped by category.

        Leaves session.dataframe completely unchanged.
        Sets the box plot on session.pending_figure.

        Args:
            text_column: The column containing document text. Defaults to "text".
            category_column: The column containing category labels. Defaults to "category_name".

        Returns:
            Dict containing overall text length stats and per-category stats.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session state."}
        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        # Work on a copy to ensure session.dataframe remains untouched
        df = session.dataframe.copy()
        df["text_length"] = df[text_column].fillna("").apply(len)

        # Helper to format .describe() series into a clean dict
        def format_describe(series):
            desc = series.describe()
            return {
                "count": int(desc["count"]),
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4),
                "min": float(desc["min"]),
                "25%": round(float(desc["25%"]), 4),
                "50%": round(float(desc["50%"]), 4),
                "75%": round(float(desc["75%"]), 4),
                "max": float(desc["max"])
            }

        overall_stats = format_describe(df["text_length"])

        per_category_stats = {}
        if category_column in df.columns:
            for cat, group in df.groupby(category_column):
                per_category_stats[str(cat)] = format_describe(group["text_length"])

        # Build Seaborn/Matplotlib box plot
        fig, ax = plt.subplots(figsize=(8, 6))

        if category_column in df.columns:
            sns.boxplot(data=df, x=category_column, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=df["text_length"], ax=ax)
            ax.set_xlabel("Dataset")

        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution by Category")
        plt.tight_layout()

        # Set pending_figure plot requirement
        session.pending_figure = fig

        res_id = session.next_result_id("describe_data")
        args = {"text_column": text_column, "category_column": category_column}

        summary = {
            "result_id": res_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", args=args, summary=summary)

        return summary

    return [check_missing_tool, check_duplicates_tool, describe_data_tool]
