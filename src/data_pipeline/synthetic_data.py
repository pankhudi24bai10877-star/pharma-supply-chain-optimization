import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(
    n_days=365,
    mean_demand=100,
    demand_dispersion=10,
    disruption_probability=0.05,
    lead_time_median=4,
    lead_time_sigma=0.5,
    seed=42
):
    """
    Generate synthetic pharmaceutical supply-chain data.

    Demand:
        Negative Binomial distribution

    Disruptions:
        Bernoulli distribution

    Lead Time:
        Log-normal distribution
    """

    np.random.seed(seed)

    # -----------------------------
    # 1. Demand - Negative Binomial
    # -----------------------------
    # Negative Binomial parameters
    # variance = mean + mean^2 / r
    r = demand_dispersion
    p = r / (r + mean_demand)

    demand = np.random.negative_binomial(r, p, n_days)

    # -----------------------------
    # 2. Disruption - Bernoulli
    # -----------------------------
    disruption = np.random.binomial(
        n=1,
        p=disruption_probability,
        size=n_days
    )

    # -----------------------------
    # 3. Lead Time - Log-normal
    # -----------------------------
    # Convert median lead time to log-space
    mu = np.log(lead_time_median)

    lead_time = np.random.lognormal(
        mean=mu,
        sigma=lead_time_sigma,
        size=n_days
    )

    # Convert to realistic whole-number days
    lead_time = np.maximum(1, np.round(lead_time).astype(int))

    # -----------------------------
    # Create dataset
    # -----------------------------
    dates = pd.date_range(
        start="2026-01-01",
        periods=n_days,
        freq="D"
    )

    df = pd.DataFrame({
        "date": dates,
        "demand": demand,
        "disruption": disruption,
        "lead_time_days": lead_time
    })

    return df


def save_data(df):
    """Save generated data to the project's data directory."""

    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "synthetic_supply_chain.csv"

    df.to_csv(output_path, index=False)

    print(f"Data saved to: {output_path}")
    print(f"Number of records: {len(df)}")


if __name__ == "__main__":

    df = generate_synthetic_data()

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nDataset statistics:")
    print(df.describe())

    print("\nDisruption distribution:")
    print(df["disruption"].value_counts())

    save_data(df)