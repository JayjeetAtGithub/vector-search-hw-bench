import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from scipy.optimize import curve_fit

# Load the file
with open("data.txt", "r") as file:
    content = file.read()

# Split blocks using info header as a delimiter
blocks = content.strip().split("[INFO] Read in file")

def extract_fp_value(event_name, text):
    pattern = rf'([\d,]+)\s+{re.escape(event_name)}'
    match = re.search(pattern, text)
    return int(match.group(1).replace(',', '')) if match else 0

results = []

for block in blocks:
    if "search" not in block or "Performance counter stats" not in block:
        continue

    ds_match = re.search(r'cpu_hnsw_deep_(\d+)l', block)
    dataset_size = int(ds_match.group(1)) if ds_match else 0

    search_times = [int(m.group(1)) for m in re.finditer(r',(\d+) us', block)]
    avg_search_time_ms = mean(search_times) / 1000.0 if search_times else 0

    read_mib = float(re.search(r'([\d,\.]+) MiB\s+uncore_imc/cas_count_read', block).group(1).replace(',', ''))
    write_mib = float(re.search(r'([\d,\.]+) MiB\s+uncore_imc/cas_count_write', block).group(1).replace(',', ''))

    sp_scalar = extract_fp_value('fp_arith_inst_retired.scalar_single', block)
    sp_128b_packed = extract_fp_value('fp_arith_inst_retired.128b_packed_single', block)
    sp_256b_packed = extract_fp_value('fp_arith_inst_retired.256b_packed_single', block)
    sp_512b_packed = extract_fp_value('fp_arith_inst_retired.512b_packed_single', block)

    clk_match = re.search(r'([\d,]+)\s+cpu_clk_unhalted\.thread', block)
    inst_match = re.search(r'([\d,]+)\s+inst_retired\.any', block)
    if clk_match and inst_match:
        clk = int(clk_match.group(1).replace(',', ''))
        inst = int(inst_match.group(1).replace(',', ''))
        ipc = inst / clk if clk else 0
    else:
        ipc = 0

    results.append({
        "dataset_size": dataset_size,
        "avg_search_time_ms": avg_search_time_ms,
        "read_mib": read_mib,
        "write_mib": write_mib,
        "sp_scalar": sp_scalar,
        "sp_128b_packed": sp_128b_packed,
        "sp_256b_packed": sp_256b_packed,
        "sp_512b_packed": sp_512b_packed,
        "ipc": ipc
    })

df = pd.DataFrame(results)
df.to_csv("perf_stats.csv", index=False)

def power_law(x, a, b):
    return a * np.power(x, b)

df = pd.read_csv("perf_stats.csv")

# Derived metrics
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

x = df['dataset_size'].values

fig, axs = plt.subplots(2, 3, figsize=(18, 8))
fig.tight_layout(pad=4.0)
axs = axs.flatten()

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

# ---- Subplot 4: Operational Intensity ----
y_oi = df['operational_intensity'].values
popt_oi, _ = curve_fit(power_law, x, y_oi)
axs[3].plot(x, y_oi, marker='o', linestyle='-', color='purple', label='Op. Intensity')
axs[3].plot(x, power_law(x, *popt_oi), 'k--',
            label=rf'Trend $y={popt_oi[0]:.2e} \cdot x^{{{popt_oi[1]:.2f}}}$')

axs[3].set_xlabel("Dataset Size")
axs[3].set_ylabel("Operational Intensity (FLOP/Byte)")
axs[3].grid()
axs[3].legend()

# ---- Subplot 5: IPC ----
y_ipc = df['ipc'].values
popt_ipc, _ = curve_fit(power_law, x, y_ipc)
axs[4].plot(x, y_ipc, marker='o', linestyle='-', color='brown', label='IPC')
axs[4].plot(x, power_law(x, *popt_ipc), 'k--',
            label=rf'Trend $y={popt_ipc[0]:.2e} \cdot x^{{{popt_ipc[1]:.2f}}}$')

axs[4].set_xlabel("Dataset Size")
axs[4].set_ylabel("IPC")
axs[4].grid()
axs[4].legend()

# ---- Remove unused subplot (axs[5]) ----
fig.delaxes(axs[5])

# Save combined figure
plt.savefig("perf_stats.pdf")
