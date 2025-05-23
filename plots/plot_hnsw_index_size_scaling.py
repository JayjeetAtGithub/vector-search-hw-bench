import re
import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the file
with open('hnsw_index_size_scaling', 'r') as f:
    raw_data = f.read()

# Parse the data
data = []
for line in raw_data.strip().split('\n'):
    size_str, filename = line.strip().split()
    match = re.search(r'_(\d+)M\.faiss', filename)
    config_str = match.group(1) if match else None

    # Convert size to MiB
    if size_str.endswith('M'):
        size_mib = float(size_str[:-1])
    elif size_str.endswith('G'):
        size_mib = float(size_str[:-1]) * 1024
    else:
        continue  # Unknown format

    data.append((config_str, size_mib))

# Create DataFrame
df = pd.DataFrame(data, columns=['config_label', 'size_MiB'])

# Plot using index for even spacing
plt.figure(figsize=(5, 2))
plt.plot(range(len(df)), df['size_MiB'], marker='o')
plt.xticks(ticks=range(len(df)), labels=df['config_label'], rotation=45)
plt.xlabel('HNSW Parameter (M)')
plt.ylabel('Index Size (MiB)')
plt.grid(True)
plt.tight_layout(pad=1.0)
plt.savefig('hnsw_index_size_scaling.pdf')