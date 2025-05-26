import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("core_scaling_hnsw.csv")

# Filter for index sizes 1M and 32M
filtered_df = df[df['index_size_million'].isin([1.0, 32.0])]

# Compute average duration for each core count and index size
grouped = filtered_df.groupby(['core', 'index_size_million'])['duration_us'].mean().reset_index()
grouped['duration_s'] = grouped['duration_us'] / 1e6
grouped['qps_s'] = 100000 / grouped['duration_s']

# Pivot table for plotting
pivot_df = grouped.pivot(index='core', columns='index_size_million', values='qps_s')

# Prepare evenly spaced x-axis
core_counts = sorted(pivot_df.index)
positions = range(len(core_counts))

# Plot
markers = ['o', 'x', '^', 'D', 'x', 'v']
plt.figure(figsize=(5, 2.5))
for i, index_size in enumerate(pivot_df.columns):
    values = [pivot_df.loc[core, index_size] for core in core_counts]
    plt.plot(positions, values, marker=markers[i], label=f'{int(index_size)}M')

plt.xlabel('CPU Core Count')
plt.ylabel('QPS')
plt.xticks(positions, labels=core_counts)
plt.legend(title='Index Size')
plt.grid(True)
plt.tight_layout(pad=1.0)
plt.savefig('core_scaling_hnsw.pdf')
