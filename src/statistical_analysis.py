from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from common import CONFIGURATIONS, METRIC_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
OUTPUT = ROOT / "outputs" / "statistics"


def load_results() -> pd.DataFrame:
    classical = pd.read_csv(METRICS_DIR / "classical_model_metrics.csv")
    deep = pd.read_csv(METRICS_DIR / "deep_model_metrics.csv")
    return pd.concat([classical, deep], ignore_index=True)


def rank_biserial_from_differences(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    denominator = positive + negative
    return float((positive - negative) / denominator)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = load_results()

    key_columns = ["Model Family", "Algorithm"]
    pivoted = {
        metric: results.pivot(
            index=key_columns,
            columns="Configuration",
            values=metric,
        ).loc[:, CONFIGURATIONS]
        for metric in METRIC_COLUMNS
    }

    descriptive_rows = []
    family_rows = []
    mean_change_rows = []
    frequency_rows = []
    friedman_rows = []
    rank_rows = []
    wilcoxon_rows = []

    for metric, table in pivoted.items():
        for configuration in CONFIGURATIONS:
            values = table[configuration].to_numpy()
            descriptive_rows.append(
                {
                    "Metric": metric,
                    "Configuration": configuration,
                    "Mean": values.mean(),
                    "Median": np.median(values),
                    "Std. Dev.": values.std(ddof=1),
                    "Minimum": values.min(),
                    "Maximum": values.max(),
                    "Range": values.max() - values.min(),
                }
            )

            for family in ["Classical ML", "Deep Learning"]:
                family_values = table.xs(family, level="Model Family")[
                    configuration
                ].to_numpy()
                family_rows.append(
                    {
                        "Metric": metric,
                        "Configuration": configuration,
                        "Model Family": family,
                        "Mean": family_values.mean(),
                        "Median": np.median(family_values),
                        "Std. Dev.": family_values.std(ddof=1),
                        "Minimum": family_values.min(),
                        "Maximum": family_values.max(),
                    }
                )

        baseline = table["Baseline"]
        for configuration in CONFIGURATIONS[1:]:
            differences = table[configuration] - baseline
            mean_change_rows.append(
                {
                    "Metric": metric,
                    "Enhanced Configuration": configuration,
                    "All Models": differences.mean(),
                    "Classical ML": differences.xs(
                        "Classical ML",
                        level="Model Family",
                    ).mean(),
                    "Deep Learning": differences.xs(
                        "Deep Learning",
                        level="Model Family",
                    ).mean(),
                }
            )
            frequency_rows.append(
                {
                    "Metric": metric,
                    "Enhanced Configuration": configuration,
                    "Improved Models": int((differences > 0).sum()),
                    "Unchanged Models": int((differences == 0).sum()),
                    "Decreased Models": int((differences < 0).sum()),
                }
            )

        statistic, p_value = friedmanchisquare(
            *[table[configuration].to_numpy() for configuration in CONFIGURATIONS]
        )
        n_models = len(table)
        kendalls_w = statistic / (n_models * (len(CONFIGURATIONS) - 1))
        friedman_rows.append(
            {
                "Metric": metric,
                "Models": n_models,
                "Friedman Chi-Square": statistic,
                "p-Value": p_value,
                "Kendall's W": kendalls_w,
            }
        )

        row_ranks = table.apply(
            lambda row: rankdata(-row.to_numpy(), method="average"),
            axis=1,
            result_type="expand",
        )
        row_ranks.columns = CONFIGURATIONS
        mean_ranks = row_ranks.mean(axis=0)
        rank_rows.append(
            {
                "Metric": metric,
                **{configuration: mean_ranks[configuration] for configuration in CONFIGURATIONS},
            }
        )

        metric_pair_rows = []
        raw_p_values = []

        for configuration_1, configuration_2 in combinations(CONFIGURATIONS, 2):
            differences = (
                table[configuration_2] - table[configuration_1]
            ).to_numpy()

            statistic, raw_p = wilcoxon(
                table[configuration_1],
                table[configuration_2],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            raw_p_values.append(raw_p)
            metric_pair_rows.append(
                {
                    "Metric": metric,
                    "Configuration 1": configuration_1,
                    "Configuration 2": configuration_2,
                    "Mean Difference": differences.mean(),
                    "Wilcoxon W": statistic,
                    "Raw p-Value": raw_p,
                    "Rank-Biserial r": rank_biserial_from_differences(differences),
                }
            )

        adjusted = multipletests(raw_p_values, method="holm")[1]
        for row, adjusted_p in zip(metric_pair_rows, adjusted):
            row["Holm-Adjusted p-Value"] = adjusted_p
            wilcoxon_rows.append(row)

    best_rows = []
    for configuration in CONFIGURATIONS:
        subset = results[results["Configuration"] == configuration]
        best_rows.append(
            subset.loc[subset["Accuracy"].idxmax()].to_dict()
        )

    pd.DataFrame(descriptive_rows).to_csv(
        OUTPUT / "descriptive_statistics_all_metrics.csv",
        index=False,
    )
    pd.DataFrame(family_rows).to_csv(
        OUTPUT / "descriptive_statistics_by_family.csv",
        index=False,
    )
    pd.DataFrame(mean_change_rows).to_csv(
        OUTPUT / "mean_metric_changes.csv",
        index=False,
    )
    pd.DataFrame(frequency_rows).to_csv(
        OUTPUT / "improvement_frequencies.csv",
        index=False,
    )
    pd.DataFrame(friedman_rows).to_csv(
        OUTPUT / "friedman_tests.csv",
        index=False,
    )
    pd.DataFrame(rank_rows).to_csv(
        OUTPUT / "mean_ranks.csv",
        index=False,
    )
    pd.DataFrame(wilcoxon_rows).to_csv(
        OUTPUT / "wilcoxon_holm_tests.csv",
        index=False,
    )
    pd.DataFrame(best_rows).to_csv(
        OUTPUT / "best_model_per_configuration.csv",
        index=False,
    )

    with pd.ExcelWriter(
        OUTPUT / "complete_statistical_analysis.xlsx",
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(descriptive_rows).to_excel(
            writer,
            sheet_name="Descriptive",
            index=False,
        )
        pd.DataFrame(family_rows).to_excel(
            writer,
            sheet_name="By Family",
            index=False,
        )
        pd.DataFrame(mean_change_rows).to_excel(
            writer,
            sheet_name="Mean Changes",
            index=False,
        )
        pd.DataFrame(frequency_rows).to_excel(
            writer,
            sheet_name="Frequencies",
            index=False,
        )
        pd.DataFrame(friedman_rows).to_excel(
            writer,
            sheet_name="Friedman",
            index=False,
        )
        pd.DataFrame(rank_rows).to_excel(
            writer,
            sheet_name="Mean Ranks",
            index=False,
        )
        pd.DataFrame(wilcoxon_rows).to_excel(
            writer,
            sheet_name="Wilcoxon Holm",
            index=False,
        )
        pd.DataFrame(best_rows).to_excel(
            writer,
            sheet_name="Best Models",
            index=False,
        )

    print(f"Statistical outputs written to: {OUTPUT}")


if __name__ == "__main__":
    main()
