import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * np.power(x, b)

if __name__ == "__main__":
    df = pd.read_csv("perf_data.csv")
    df.rename(columns={'dataset_size_l': 'dataset_size'}, inplace=True)

    # Compute derived metrics
    df['read_bytes'] = df['read_mib'] * (1024 * 1024)
    df['write_bytes'] = df['write_mib'] * (1024 * 1024)
    df['tota_read_bytes_per_iteration'] = df['read_bytes'] / 20
    df['total_write_bytes_per_iteration'] = df['write_bytes'] / 20
    df['avg_read_bytes_per_query'] = df['tota_read_bytes_per_iteration'] / 10000
    df['avg_write_bytes_per_query'] = df['total_write_bytes_per_iteration'] / 10000
    df['avg_read_mib_per_query'] = df['avg_read_bytes_per_query'] / (1024 * 1024)
    df['avg_write_mib_per_query'] = df['avg_write_bytes_per_query'] / (1024 * 1024)
    df['avg_total_mib_per_query'] = df['avg_read_mib_per_query'] + df['avg_write_mib_per_query']
    df['sp_flops'] = df['sp_scalar'] + df['sp_128b_packed'] * 4 + df['sp_256b_packed'] * 8 + df['sp_512b_packed'] * 16
    df['sp_gflops'] = df['sp_flops'] / (10**9)

    df['operational_intensity'] = df['sp_flops'] / (df['read_bytes'])
    print("Operational Intensity (FLOP/Byte):")
    print(df['dataset_size'], df['operational_intensity'])
    print(df)

    x = df['dataset_size'].values

    fig, axs = plt.subplots(1, 3, figsize=(14, 3))
    fig.tight_layout(pad=2.5)

    # ---- Subplot 1: MiB per Query ----
    y_read = df['avg_read_mib_per_query'].values
    y_write = df['avg_write_mib_per_query'].values
    y_total = df['avg_total_mib_per_query'].values

    popt_read, _ = curve_fit(power_law, x, y_read)
    popt_write, _ = curve_fit(power_law, x, y_write)
    popt_total, _ = curve_fit(power_law, x, y_total)

    axs[0].plot(x, y_read, marker='o', linestyle='-', color='b',
                label=rf'Read $y={popt_read[0]:.2e} \cdot x^{{{popt_read[1]:.2f}}}$')
    axs[0].plot(x, y_write, marker='^', linestyle='-', color='r',
                label=rf'Write $y={popt_write[0]:.2e} \cdot x^{{{popt_write[1]:.2f}}}$')
    axs[0].plot(x, y_total, marker='X', linestyle='-', color='g',
                label=rf'Total $y={popt_total[0]:.2e} \cdot x^{{{popt_total[1]:.2f}}}$')

    axs[0].plot(x, power_law(x, *popt_read), 'b--')
    axs[0].plot(x, power_law(x, *popt_write), 'r--')
    axs[0].plot(x, power_law(x, *popt_total), 'g--')

    axs[0].set_xlabel("Dataset Size")
    axs[0].set_ylabel("MiB / Query")
    axs[0].grid()
    axs[0].legend()

    # ---- Subplot 2: GFLOPS ----
    y_gflops = df['sp_gflops'].values
    popt_gflops, _ = curve_fit(power_law, x, y_gflops)
    axs[1].plot(x, y_gflops, marker='o', linestyle='-', color='orange', label='GFLOPS')
    axs[1].plot(x, power_law(x, *popt_gflops), 'k--',
                label=rf'Trend $y={popt_gflops[0]:.2e} \cdot x^{{{popt_gflops[1]:.2f}}}$')

    axs[1].set_xlabel("Dataset Size")
    axs[1].set_ylabel("GFLOPS")
    axs[1].grid()
    axs[1].legend()

    # ---- Subplot 3: Avg. Search Time ----
    y_search = df['avg_search_time_ms'].values
    popt_time, _ = curve_fit(power_law, x, y_search)
    axs[2].plot(x, y_search, marker='o', linestyle='-', color='green', label='Search Time (ms)')
    axs[2].plot(x, power_law(x, *popt_time), 'k--',
                label=rf'Trend $y={popt_time[0]:.2e} \cdot x^{{{popt_time[1]:.2f}}}$')

    axs[2].set_xlabel("Dataset Size")
    axs[2].set_ylabel("Search Duration (ms)")
    axs[2].grid()
    axs[2].legend()

    # Save combined figure
    plt.savefig("combined_plots.pdf")
