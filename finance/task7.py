import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


pd.set_option('display.expand_frame_repr', False)


"""
Try to find a portfolio of S&P500 stocks beating the market.

1. Get list of SP500 stocks from CSV file
2. Pick a time period and download data
3. Perform hierarchical clustering with different methods and distances
4. Compare to SP500 index
5. Isolate clusters beating the market
"""


# ============================================================
# 1. LOAD S&P 500 TICKERS
# ============================================================

tickers = pd.read_csv('sp500.csv')

stocks = tickers.iloc[:, 1].to_list()

# Add S&P 500 benchmark
stocks_with_benchmark = stocks + ["^GSPC"]


# Remove problematic tickers
for ticker in ['EA', 'AVB', 'EQR', 'DAY', 'CTRA', 'HOLX', 'BK']:
    if ticker in stocks_with_benchmark:
        stocks_with_benchmark.remove(ticker)


# ============================================================
# 2. DOWNLOAD DATA
# ============================================================

stocks_data = yf.download(
    stocks_with_benchmark,
    start='2025-01-01',
    end='2026-01-01',
    auto_adjust=False
)['Adj Close']


# ============================================================
# 3. CLEAN DATA
# ============================================================

# Remove columns that contain only missing values
stocks_data = stocks_data.dropna(axis=1, how='all')

# Forward fill missing prices
stocks_data = stocks_data.ffill()

# Remove rows where everything is missing
stocks_data = stocks_data.dropna(axis=0, how='all')


# Make sure benchmark exists
if "^GSPC" not in stocks_data.columns:
    raise ValueError("S&P 500 (^GSPC) data was not downloaded.")


# ============================================================
# 4. CALCULATE RETURNS
# ============================================================

returns = stocks_data.pct_change()

# Remove rows where all stocks have missing returns
returns = returns.dropna(axis=0, how='all')


# Benchmark
benchmark_returns = returns["^GSPC"].dropna()

# Only actual stocks, not S&P 500
stock_returns = returns.drop(columns="^GSPC")


# Remove stocks that still contain missing values
stock_returns = stock_returns.dropna(axis=1)


# Make sure benchmark and stocks use same dates
common_dates = stock_returns.index.intersection(
    benchmark_returns.index
)

stock_returns = stock_returns.loc[common_dates]
benchmark_returns = benchmark_returns.loc[common_dates]


# ============================================================
# 5. CORRELATION MATRIX
# ============================================================

correlation_matrix = stock_returns.corr()

print("\nCorrelation matrix:")
print(correlation_matrix)


# ============================================================
# 6. DISTANCE MATRIX
# ============================================================

# Correlation distance:
# correlation = 1    -> distance = 0
# correlation = 0    -> distance = 1
# correlation = -1   -> distance = 2

distance_matrix = (
    1 - correlation_matrix
).to_numpy(copy=True)


# Diagonal must be zero
np.fill_diagonal(distance_matrix, 0)


# Convert square distance matrix to condensed form
condensed_distance_matrix = squareform(
    distance_matrix
)


# ============================================================
# 7. CLUSTERING SETTINGS
# ============================================================

methods = [
    'single',
    'complete',
    'average',
    'centroid',
    'ward',
    'median'
]

max_d_values = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4
]


# ============================================================
# 8. PLOT SETUP
# ============================================================

fig, axes = plt.subplots(
    len(methods),
    len(max_d_values),
    figsize=(40, 20),
    sharex=True,
    sharey=True
)

fig.suptitle(
    "Cluster-Based Portfolio vs Benchmark",
    fontsize=20
)


# ============================================================
# 9. STORE ALL CLUSTER RESULTS
# ============================================================

all_cluster_results = []


# S&P 500 total return
benchmark_total_return = (
    1 + benchmark_returns
).prod() - 1


# ============================================================
# 10. CLUSTERING
# ============================================================

for i, method in enumerate(methods):

    # Create hierarchical clustering
    linkage_matrix = linkage(
        condensed_distance_matrix,
        method=method
    )


    for j, max_d in enumerate(max_d_values):

        # Cut dendrogram into clusters
        clusters = fcluster(
            linkage_matrix,
            max_d,
            criterion='distance'
        )


        # ====================================================
        # CALCULATE EACH CLUSTER
        # ====================================================

        cluster_results = []

        unique_clusters = np.unique(clusters)


        for cluster_id in unique_clusters:

            # Stocks belonging to this cluster
            cluster_stocks = stock_returns.columns[
                clusters == cluster_id
            ]


            # Equal-weighted portfolio
            cluster_return = stock_returns[
                cluster_stocks
            ].mean(axis=1)


            # Cumulative return
            cluster_total_return = (
                1 + cluster_return
            ).prod() - 1


            # Store cluster information
            cluster_result = {
                "Method": method,
                "Max Distance": max_d,
                "Cluster": cluster_id,
                "Stocks": list(cluster_stocks),
                "Number of stocks": len(cluster_stocks),
                "Return": cluster_total_return
            }


            cluster_results.append(
                cluster_result
            )

            all_cluster_results.append(
                cluster_result
            )


        # ====================================================
        # CREATE PORTFOLIO FROM ALL CLUSTERS
        # ====================================================

        cluster_returns = []

        for cluster_id in unique_clusters:

            cluster_stocks = stock_returns.columns[
                clusters == cluster_id
            ]

            cluster_data = stock_returns[
                cluster_stocks
            ].mean(axis=1)

            cluster_returns.append(
                cluster_data
            )


        # Equal-weighted portfolio of clusters
        combined_returns = pd.concat(
            cluster_returns,
            axis=1
        ).mean(axis=1)


        # Cumulative portfolio return
        combined_cum_returns = (
            1 + combined_returns
        ).cumprod() - 1


        # Cumulative S&P 500 return
        benchmark_cum_returns = (
            1 + benchmark_returns
        ).cumprod() - 1


        # ====================================================
        # PLOT
        # ====================================================

        ax = axes[i, j]

        combined_cum_returns.plot(
            ax=ax,
            color="blue"
        )

        benchmark_cum_returns.plot(
            ax=ax,
            label="S&P 500",
            color="orange"
        )

        ax.set_title(
            f"{method}, max_d={max_d}"
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")

        ax.legend()


# ============================================================
# 11. SHOW PLOTS
# ============================================================

plt.tight_layout(
    rect=[0, 0.03, 1, 0.95]
)

plt.show()


# ============================================================
# 12. ALL CLUSTER RESULTS INTO DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_cluster_results
)


# ============================================================
# 13. COMPARE EACH CLUSTER TO S&P 500
# ============================================================

results_df["S&P 500 Return"] = (
    benchmark_total_return
)

results_df["Excess Return"] = (
    results_df["Return"]
    - results_df["S&P 500 Return"]
)

results_df["Beats S&P 500"] = (
    results_df["Return"]
    > results_df["S&P 500 Return"]
)


# ============================================================
# 14. PRINT ALL RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("ALL CLUSTER RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "Method",
            "Max Distance",
            "Cluster",
            "Number of stocks",
            "Return",
            "S&P 500 Return",
            "Excess Return",
            "Beats S&P 500"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. ISOLATE CLUSTERS BEATING THE MARKET
# ============================================================

beating_clusters = results_df[
    results_df["Beats S&P 500"]
].copy()


print("\n")
print("=" * 100)
print("CLUSTERS BEATING S&P 500")
print("=" * 100)


if len(beating_clusters) == 0:

    print("No clusters beat the S&P 500.")

else:

    print(
        beating_clusters[
            [
                "Method",
                "Max Distance",
                "Cluster",
                "Number of stocks",
                "Return",
                "S&P 500 Return",
                "Excess Return"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 16. PRINT STOCKS IN WINNING CLUSTERS
# ============================================================

print("\n")
print("=" * 100)
print("STOCKS IN CLUSTERS BEATING S&P 500")
print("=" * 100)


for _, row in beating_clusters.iterrows():

    print("\n" + "-" * 100)

    print(
        f"Method: {row['Method']}"
    )

    print(
        f"Max Distance: {row['Max Distance']}"
    )

    print(
        f"Cluster: {row['Cluster']}"
    )

    print(
        f"Number of stocks: {row['Number of stocks']}"
    )

    print(
        f"Cluster return: {row['Return']:.2%}"
    )

    print(
        f"S&P 500 return: {row['S&P 500 Return']:.2%}"
    )

    print(
        f"Excess return: {row['Excess Return']:.2%}"
    )

    print("Stocks:")

    print(
        ", ".join(row["Stocks"])
    )