import re
import matplotlib.pyplot as plt

# Path to the file
filename = "data.txt"

# Read file
with open(filename, "r") as f:
    raw_data = f.read()

# Extract all data
core_pattern = r"(\d+)\s+cores"
read_pattern = r"Read bandwidth:\s+([\d.]+)"
write_pattern = r"Write bandwidth:\s+([\d.]+)"
total_pattern = r"Total bandwidth:\s+([\d.]+)"

all_cores = [int(x) for x in re.findall(core_pattern, raw_data)]
all_read_bw = [float(x) for x in re.findall(read_pattern, raw_data)]
all_write_bw = [float(x) for x in re.findall(write_pattern, raw_data)]
all_total_bw = [float(x) for x in re.findall(total_pattern, raw_data)]

# Filter for desired cores
target_cores = [1, 2, 4, 8, 16, 24]
filtered_cores = []
filtered_read_bw = []
filtered_write_bw = []
filtered_total_bw = []

for core, r, w, t in zip(all_cores, all_read_bw, all_write_bw, all_total_bw):
    if core in target_cores:
        filtered_cores.append(core)
        filtered_read_bw.append(r * 1.07 * 1000)
        filtered_write_bw.append(w * 1.07 * 1000)
        filtered_total_bw.append(t * 1.07 * 1000)

# Print filtered data
print("Cores\tRead BW\t\tWrite BW\tTotal BW")
for c, r, w, t in zip(filtered_cores, filtered_read_bw, filtered_write_bw, filtered_total_bw):
    print(f"{c}\t{r:.4f} GB/s\t{w:.4f} GB/s\t{t:.4f} GB/s")

# Plot
plt.figure(figsize=(5, 3))
plt.plot(filtered_cores, filtered_total_bw, marker='o', label='Total B/W')
plt.plot(filtered_cores, filtered_read_bw, marker='^', label='Read B/W')
plt.plot(filtered_cores, filtered_write_bw, marker='x', label='Write B/W')

plt.xticks(target_cores)
plt.xlabel('# of Cores')
plt.ylabel('Bandwidth (MB/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('triad_mem.pdf')
