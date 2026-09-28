"""Generate and summarize the baseline simulated dataset."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIRECTORY))

from simulation import simulate_customers  # noqa: E402


def main() -> None:
    output_directory = PROJECT_ROOT / "data"
    output_directory.mkdir(parents=True, exist_ok=True)

    data = simulate_customers(
        number_of_customers=20_000,
        seed=42,
    )

    output_path = output_directory / "baseline_customers.csv"
    data.to_csv(output_path, index=False)

    naive_effect = (
        data.loc[data["treatment"] == 1, "retained_90d"].mean()
        - data.loc[data["treatment"] == 0, "retained_90d"].mean()
    )

    true_ate = data["true_cate"].mean()
    profitable_share = (80 * data["true_cate"] - 10 > 0).mean()

    summary = pd.Series(
        {
            "number_of_customers": len(data),
            "treatment_rate": data["treatment"].mean(),
            "retention_rate": data["retained_90d"].mean(),
            "naive_effect": naive_effect,
            "true_ate": true_ate,
            "naive_bias": naive_effect - true_ate,
            "minimum_propensity": data["true_propensity"].min(),
            "maximum_propensity": data["true_propensity"].max(),
            "cate_standard_deviation": data["true_cate"].std(),
            "profitable_customer_share": profitable_share,
        }
    )

    print(f"Dataset written to: {output_path}")
    print()
    print(summary.to_string())


if __name__ == "__main__":
    main()
