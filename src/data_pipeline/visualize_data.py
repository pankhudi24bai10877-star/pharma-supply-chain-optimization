import pandas as pd
import matplotlib.pyplot as plt


def visualize_data(file_path="data/synthetic_supply_chain.csv"):
    df = pd.read_csv(file_path)

    # Demand
    plt.figure(figsize=(10, 5))
    plt.hist(df["demand"], bins=30)
    plt.title("Medicine Demand Distribution")
    plt.xlabel("Demand (units)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Disruption
    plt.figure(figsize=(6, 5))
    df["disruption"].value_counts().sort_index().plot(kind="bar")
    plt.title("Supply Chain Disruptions")
    plt.xlabel("Disruption (0 = No, 1 = Yes)")
    plt.ylabel("Number of Days")
    plt.xticks([0, 1], ["No Disruption", "Disruption"], rotation=0)
    plt.tight_layout()

    # Lead time
    plt.figure(figsize=(10, 5))
    plt.hist(df["lead_time_days"], bins=20)
    plt.title("Lead Time Distribution")
    plt.xlabel("Lead Time (days)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Show all graphs
    plt.show()


if __name__ == "__main__":
    visualize_data()