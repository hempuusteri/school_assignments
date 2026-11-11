import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
pd.set_option('display.expand_frame_repr', False)

"""

Try to find a portfolio of S&P500 stocks beating the market.

   1. Get list of SP500 stocks from CSV file
   2. Pick a time period of your choice and download data. 
    You may first use just a portion of all possible tickers.
   3. Perform hierarchical clustering with different methods and distances
   4. Compare to SP500 index
   5. Isolate clusters beating the market
"""
tickers = pd.read_csv('sp500.csv')
#stocks = tickers.iloc[:,1].to_list() + ["^GSPC"]

#stocks.remove('EA')
#stocks.remove('AVB')
#stocks.remove('EQR')
#stocks.remove('DAY')
#stocks.remove('CTRA')
#stocks.remove('HOLX')
#stocks.remove('BK')


stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'JNJ', 'V', 'WMT',
'NVDA','JPM','UNH','HD','PG','MA','DIS','PYPL','ADBE','NFLX',
'PEP','INTC','KO','NKE','MRK','PFE','T','VZ','XOM','CSCO', '^GSPC']

stocks_data = yf.download(stocks, start = '2025-01-01', end = '2026-01-01',auto_adjust=False)['Adj Close']

returns = stocks_data.pct_change().dropna()
benchmark_returns = returns["^GSPC"]
stock_returns = returns.drop(columns="^GSPC")

scaler = StandardScaler()
normalized_returns = pd.DataFrame(scaler.fit_transform(returns), columns=returns.columns, index=returns.index)
correlation_matrix = normalized_returns.corr()

methods = ['single', 'complete', 'average', 'centroid', 'ward']
max_d_values = [0.1, 0.4, 0.6, 0.7, 0.8, 1.0, 1.3]

fig, axes = plt.subplots(len(methods), len(max_d_values), figsize=(50, 30), sharex=True, sharey=True)
fig.suptitle("Cluster-Based Portfolio vs Benchmark", fontsize=16)

distance_matrix = 1 - correlation_matrix.abs()
condensed_distance_matrix = pdist(distance_matrix)

all_cluster_returns = []

for i, method in enumerate(methods):
    linkage_matrix = linkage(condensed_distance_matrix, method=method)

    for j, max_d in enumerate(max_d_values):
        clusters = fcluster(linkage_matrix, max_d, criterion='distance')

        # Cluster-based portfolio
        cluster_returns = []

        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            cluster_stocks = [stocks[k] for k in range(len(stocks)) if clusters[k] == cluster]
            cluster_data = returns[cluster_stocks].mean(axis=1)
            cluster_returns.append(cluster_data)

            cluster_total_return = (1 + cluster_data).prod() -1

            all_cluster_returns.append({"Method": method, "Max Distance":max_d, "Cluster": cluster,
                                        "Number of stocks":len(cluster_stocks), "Stocks": list(cluster_stocks), 
                                        "Return": cluster_total_return})


        # Combine cluster returns into a portfolio
        combined_returns = pd.concat(cluster_returns, axis=1).mean(axis=1)
        combined_cum_returns = (1 + combined_returns).cumprod() - 1
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1

        # Plot
        ax = axes[i, j]
        combined_cum_returns.plot(ax=ax, color="blue")
        benchmark_cum_returns.plot(ax=ax, label="S&P 500", color="orange")
        ax.set_title(f"{method}, max_d={max_d}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('kuvaaja.png')
plt.show()

benchmark_total_return = (1 + benchmark_returns).prod() -1
results_df = pd.DataFrame(all_cluster_returns)

# Clusters that beats S&P 500
beating_clusters = results_df[results_df["Return"] > benchmark_total_return]

print(f"\nS&P 500 return: "f"{benchmark_total_return:.2%}")

print("\nClusters beating the market:")

for _, row in beating_clusters.iterrows():
    stocks_in_cluster = ", ".join(row["Stocks"])

    print(f"{row['Method']:8} "
        f"{row['Max Distance']:4.1f} "
        f"cluster {row['Cluster']:3} | "
        f"{row['Number of stocks']:2} stocks | "
        f"{row['Return']:.2%} | "
        f"{stocks_in_cluster}")
