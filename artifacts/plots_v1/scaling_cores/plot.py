import pandas as pd
import matplotlib.pyplot as plt

# Load the data (or use df if already in memory)
df = pd.read_csv("data_dump.csv")  # Replace with your actual filename or use df directly

# Group by core count and compute average QPS
avg_qps = df.groupby("core_count")["qps"].mean().reset_index()

# Plot
plt.figure(figsize=(4.5, 2.5))
plt.plot(avg_qps["core_count"], avg_qps["qps"], marker='o', linestyle='-')
plt.xticks([1, 2, 4, 8, 16, 24])
plt.xlabel("# of Cores")
plt.ylabel("QPS")
plt.grid(True)
plt.tight_layout(pad=1.0)
plt.savefig('scaling_cores_plot.pdf')