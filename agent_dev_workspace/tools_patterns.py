from typing import Dict, Any, Optional
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as DF2DB
import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth


def make_tools(session):
    """Factory function returning tools for frequent pattern mining."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent word patterns for a specific category using PAMI algorithms.

        Applies per-category vectorization (with English stop words removed) and vocabulary filtering.
        Supports 'fpgrowth', 'topk', or 'maxfpgrowth' mining algorithms.

        Args:
            category_name: The category label to filter session.dataframe rows by.
            filtering_method: Filtering approach: 'variance', 'tfidf', or 'term_frequency'.
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k pattern count (required for 'topk').

        Returns:
            Dict containing summary of mined patterns, support counts, and result_id.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session state."}
        if "category_name" not in session.dataframe.columns or "text" not in session.dataframe.columns:
            return {"error": "DataFrame missing required 'category_name' or 'text' column."}

        # Algorithm argument validation
        algo_lower = algorithm.lower()
        if algo_lower in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"error": f"min_sup parameter is required for algorithm '{algorithm}'."}
        if algo_lower == "topk" and k is None:
            return {"error": "k parameter is required for algorithm 'topk'."}

        # Step 1: Filter dataframe to specified category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Fit fresh category-local vectorizer (do not drop empty text rows)
        text_series = cat_df["text"].fillna("")
        vectorizer = CountVectorizer(stop_words="english")
        dtm = vectorizer.fit_transform(text_series).toarray()
        feature_names = np.array(vectorizer.get_feature_names_out())

        if len(feature_names) == 0:
            res_id = session.next_result_id("mine_patterns")
            args = {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived vectorization/stopword removal"
            }
            session.store_result("mine_patterns_tool", args=args, summary=summary)
            return summary

        # Step 2: Apply vocabulary filtering method
        method_lower = filtering_method.lower()
        if method_lower == "variance":
            variances = np.var(dtm, axis=0, ddof=0)
            p5, p95 = np.percentile(variances, [5, 95])
            mask = (variances > p5) & (variances < p95)
        elif method_lower == "term_frequency":
            counts = np.sum(dtm, axis=0)
            p5, p95 = np.percentile(counts, [5, 95])
            mask = (counts > p5) & (counts < p95)
        elif method_lower == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(dtm).toarray()
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            mask = mean_tfidf >= p20
        else:
            return {"error": f"Unsupported filtering_method '{filtering_method}'. Choose 'variance', 'tfidf', or 'term_frequency'."}

        surviving_terms = feature_names[mask]
        surviving_dtm = dtm[:, mask]

        # Step 3: Handle zero surviving terms cleanly
        res_id = session.next_result_id("mine_patterns")
        args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }

        if len(surviving_terms) == 0:
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", args=args, summary=summary)
            return summary

        # Step 4: Convert to PAMI Transactional Database matching Master exactly
        bin_df = pd.DataFrame(surviving_dtm, columns=surviving_terms)

        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".txt") as tmp_file:
            db_path = tmp_file.name

        try:
            converter = DF2DB.DF2DB(bin_df)
            converter.convert2TransactionalDatabase(db_path, ">=", 1)

            # Mine with selected PAMI algorithm
            if algo_lower == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup)
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
            elif algo_lower == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup)
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
            elif algo_lower == "topk":
                miner = FAE.FAE(db_path, k=k)
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
            else:
                return {"error": f"Unsupported algorithm '{algorithm}'. Choose 'fpgrowth', 'topk', or 'maxfpgrowth'."}

        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

        # Parse patterns into structured list
        patterns = []
        if df_res is not None and not df_res.empty:
            for _, row in df_res.iterrows():
                p_str = str(row["Patterns"]) if "Patterns" in row else str(row.iloc[0])
                s_val = int(row["Support"]) if "Support" in row else int(row.iloc[1])
                # Split whitespace-delimited itemset string into token list
                item_list = p_str.strip().split()
                patterns.append({
                    "pattern": item_list,
                    "support": s_val
                })

        # Step 5: Store artifacts and result
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algo_lower}"
        session.artifacts[artifact_key] = patterns

        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algo_lower,
            "n_patterns": len(patterns),
            "patterns_preview": patterns[:10]
        }

        session.store_result("mine_patterns_tool", args=args, summary=summary)

        return summary

    return [mine_patterns_tool]
