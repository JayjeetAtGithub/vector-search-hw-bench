import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Full dataset
data = {
    "cores": list(range(1, 25)),
    "read_bw": [
        12.3237, 23.1527, 32.6266, 42.1281, 48.9483, 46.7672, 61.4527, 65.7366, 69.2135, 71.9491, 74.4277, 76.3838,
        77.5669, 74.2727, 78.7206, 77.8923, 80.0062, 80.84, 82.2046, 74.809, 88.21, 86.5637, 96.5464, 95.1344
    ],
    "write_bw": [
        4.10792, 7.71756, 10.8755, 14.0427, 16.3161, 15.5891, 20.4842, 21.9122, 23.0712, 23.983, 24.8092, 25.4613,
        25.8556, 24.7576, 26.2402, 25.9641, 26.6687, 26.9467, 27.4015, 24.9363, 29.4033, 28.8546, 32.1821, 31.7115
    ],
    "total_bw": [
        16.4317, 30.8702, 43.5021, 56.1708, 65.2645, 62.3563, 81.937, 87.6488, 92.2846, 95.9321, 99.237, 101.845,
        103.423, 99.0302, 104.961, 103.856, 106.675, 107.787, 109.606, 99.7454, 117.613, 115.418, 128.729, 126.846
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Core counts to select
desired_cores = [1, 2, 4, 8, 16, 24]
filtered_df = df[df["cores"].isin(desired_cores)]

# Fit 2nd-degree polynomial curves
fit_read = np.polyfit(filtered_df["cores"], filtered_df["read_bw"], 2)
fit_write = np.polyfit(filtered_df["cores"], filtered_df["write_bw"], 2)
fit_total = np.polyfit(filtered_df["cores"], filtered_df["total_bw"], 2)

# Evaluate fits at actual core values
read_fit_vals = np.polyval(fit_read, filtered_df["cores"])
write_fit_vals = np.polyval(fit_write, filtered_df["cores"])
total_fit_vals = np.polyval(fit_total, filtered_df["cores"])

# Use equally spaced x-axis for visual clarity
x_labels = [str(c) for c in desired_cores]
x_positions = np.arange(len(desired_cores))

# Plot
plt.figure(figsize=(5, 3))
plt.plot(x_positions, filtered_df["read_bw"], 'o-', label='Read Bandwidth (GB/s)')
plt.plot(x_positions, filtered_df["write_bw"], 's-', label='Write Bandwidth (GB/s)')
plt.plot(x_positions, filtered_df["total_bw"], '^-', label='Total Bandwidth (GB/s)')

# Configure X-axis
plt.xticks(x_positions, x_labels)
plt.xlabel("Number of Cores")
plt.ylabel("Bandwidth (GB/s)")
plt.legend()
plt.grid(True)
plt.tight_layout(pad=1.0)
plt.savefig('bw_scaling_plot.pdf')
