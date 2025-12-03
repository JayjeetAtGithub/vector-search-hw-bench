import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

if __name__ == "__main__":
    # Update the parsing logic: assign 'deep' as the dataset, and interpret '32m' as M=32
    parsed_data = []
    with open('data.txt', 'r') as f:
        raw_data = f.readlines()

    for line in raw_data:
        if line.startswith('search'):
            parts = line.split(',')
            index_file = parts[1]
            queries = int(parts[2])
            topk = int(parts[3])
            duration_us = int(parts[4].split()[0])

            # Format: cpu_hnsw_deep_128000000l_32m.faiss
            tokens = index_file.replace('.faiss', '').split('_')
            hw = tokens[0]
            algo = tokens[1]
            dataset = tokens[2]  # 'deep'
            dataset_size = int(tokens[3].rstrip('l'))  # 128000000l -> 128000000
            param_m = int(tokens[4].rstrip('m'))  # 32m -> 32

            parsed_data.append({
                'hw': hw,
                'algo': algo,
                'dataset': dataset,
                'dataset_size': dataset_size,
                'param_m': param_m,
                'batch_size': queries,
                'k': topk,
                'duration_us': duration_us * (10000 / queries),
                'duration_s': duration_us * (10000 / queries) / 1e6,
                "qps": 10000 / (duration_us * (10000 / queries) / 1e6)
            })

        df_fixed = pd.DataFrame(parsed_data)

    # aggregate by all params except 'duration_us' and duration_s'
    df = df_fixed.groupby(['hw', 'algo', 'dataset', 'dataset_size', 'param_m', 'batch_size', 'k']).agg({
        'duration_us': 'mean',
        'duration_s': 'mean',
        'qps': 'mean'
    }).reset_index()

    df.to_csv('parsed_data.csv', index=False)

    # Load CSV
    df = pd.read_csv("parsed_data.csv")

    # Ensure columns are numeric
    df["dataset_size"] = pd.to_numeric(df["dataset_size"])
    df["duration_s"] = pd.to_numeric(df["duration_s"])

    # Power law function
    def power_law(x, a, b):
        return a * x ** b

    # ---- PLOT 1: QPS vs Dataset Size ----
    plt.figure(figsize=(5, 3))
    for batch_size in sorted(df["batch_size"].unique()):
        batch_df = df[df["batch_size"] == batch_size]
        plt.plot(batch_df["dataset_size"], batch_df["qps"], label=f"Batch {batch_size}", marker='o')
    plt.xlabel("Dataset Size (M)")
    plt.ylabel("QPS")
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('qps_scaling_batch_ds_raw.pdf')

    # ---- PLOT 2: Duration vs Dataset Size ----
    plt.figure(figsize=(5, 3))
    for batch_size in sorted(df["batch_size"].unique()):
        batch_df = df[df["batch_size"] == batch_size]
        plt.plot(batch_df["dataset_size"], batch_df["duration_s"], label=f"Batch {batch_size}", marker='o')
    plt.xlabel("Dataset Size (M)")
    plt.ylabel("Duration (s)")
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dur_scaling_batch_ds_raw.pdf')

    # ---- PLOT 3: QPS Power Law Fit ----
    plt.figure(figsize=(5, 3))
    for batch_size in sorted(df["batch_size"].unique()):
        batch_df = df[df["batch_size"] == batch_size]
        x = batch_df["dataset_size"].values
        y = batch_df["qps"].values

        if len(x) >= 2:  # Need at least 2 points to fit
            popt, _ = curve_fit(power_law, x, y)
            a, b = popt
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = power_law(x_fit, a, b)

            # plt.plot(x, y, 'o', label=f"Batch {batch_size} (data)")
            plt.plot(x_fit, y_fit, '--', label=f"BS: {batch_size}")
        else:
            plt.plot(x, y, 'o', label=f"Batch {batch_size} (insufficient data)")

    plt.xlabel("HNSW Index Size (M)")
    plt.ylabel("QPS")
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('qps_scaling_batch_ds_power_new.pdf')

    # ---- PLOT 3: Duration Power Law Fit ----
    plt.figure(figsize=(5, 3))
    for batch_size in sorted(df["batch_size"].unique()):
        batch_df = df[df["batch_size"] == batch_size]
        x = batch_df["dataset_size"].values
        y = batch_df["duration_s"].values

        if len(x) >= 2:  # Need at least 2 points to fit
            popt, _ = curve_fit(power_law, x, y)
            a, b = popt
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = power_law(x_fit, a, b)

            # plt.plot(x, y, 'o', label=f"Batch {batch_size} (data)")
            plt.plot(x_fit, y_fit, '--', label=f"Batch size: {batch_size}")
        else:
            plt.plot(x, y, 'o', label=f"Batch {batch_size} (insufficient data)")

    plt.xlabel("HNSW Index Size (M)")
    plt.ylabel("Duration (s)")
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('dur_scaling_batch_ds_power_new.pdf')

