import pandas as pd
import matplotlib.pyplot as plt

# Full dataset
cores = list(range(1, 25))
read_bandwidth = [
    11.579, 21.7194, 31.2593, 39.9144, 47.1092, 52.5884, 57.6633, 61.7851, 64.9291, 67.8612,
    69.9264, 71.9175, 73.569, 74.4508, 75.2654, 76.1386, 76.4379, 77.0864, 76.9791, 78.8507,
    82.5054, 91.7828, 93.8999, 94.8791
]
write_bandwidth = [
    3.85967, 7.2398, 10.4198, 13.3048, 15.7031, 17.5295, 19.2211, 20.595, 21.643, 22.6204,
    23.3088, 23.9725, 24.523, 24.8169, 25.0885, 25.3795, 25.4793, 25.6955, 25.6597, 26.2836,
    27.5018, 30.5943, 31.3, 31.6264
]
total_bandwidth = [
    15.4387, 28.9592, 41.6791, 53.2192, 62.8123, 70.1179, 76.8844, 82.3801, 86.5721, 90.4816,
    93.2352, 95.89, 98.092, 99.2677, 100.354, 101.518, 101.917, 102.782, 102.639, 105.134,
    110.007, 122.377, 125.2, 126.505
]

# Create DataFrame
df = pd.DataFrame({
    'Cores': cores,
    'Read Bandwidth (GiB/s)': read_bandwidth,
    'Write Bandwidth (GiB/s)': write_bandwidth,
    'Total Bandwidth (GiB/s)': total_bandwidth
})

# Filter only specific core counts
selected_cores = [1, 2, 4, 8, 16, 24]
df_filtered = df[df['Cores'].isin(selected_cores)]

# Plot
plt.figure(figsize=(5, 3))
plt.plot(df_filtered['Cores'], df_filtered['Read Bandwidth (GiB/s)'], label='Read Bandwidth', marker='o', linestyle='-')
plt.plot(df_filtered['Cores'], df_filtered['Write Bandwidth (GiB/s)'], label='Write Bandwidth', marker='s', linestyle='-')
plt.plot(df_filtered['Cores'], df_filtered['Total Bandwidth (GiB/s)'], label='Total Bandwidth', marker='^', linestyle='--')

plt.xlabel('Number of Cores')
plt.ylabel('Bandwidth (GiB/s)')
plt.xticks(selected_cores)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('triad_mem_bw_scaling.pdf')
