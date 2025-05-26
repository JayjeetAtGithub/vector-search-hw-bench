import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

def find_pcm_logs(root_dir='.'):
    pcm_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pcm_log'):
                pcm_files.append(os.path.join(dirpath, filename))
    return pcm_files

def compute_median_bandwidth(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[1:]
    return (float(df['System.4'].max().strip()), 
            float(df['System.5'].max().strip()), 
            float(df['System.6'].max().strip()))

def get_cores(filename):
    match = re.search(r'_0-(\d+)c\.pcm_log$', filename)
    if match:
        cores = int(match.group(1)) + 1
        return cores
    else:
        print("Cores not found in filename.")
        sys.exit(1)

if __name__ == '__main__':
    pcm_logs = find_pcm_logs()
    print(f"Found {len(pcm_logs)} .pcm_log files.")
    data = list()

    for log_file in pcm_logs:
        cores = get_cores(log_file)
        rd, wr, total = compute_median_bandwidth(log_file)
        data.append(
            {
                "cores": cores,
                "rd": rd,
                "wr": wr,
                "total": total,
            })

    final_df = pd.DataFrame(data)
    print(final_df)

    df_to_plot = final_df.sort_values('cores')

    # Plot
    plt.figure(figsize=(5, 2.5))
    plt.plot(df_to_plot['cores'], df_to_plot['total'], marker='o', linestyle='-', label='Total B/W')
    plt.plot(df_to_plot['cores'], df_to_plot['rd'], marker='^', linestyle='-', label='Read B/W')
    plt.plot(df_to_plot['cores'], df_to_plot['wr'], marker='X', linestyle='-', label='Write B/W')
    plt.xticks([1, 2, 4, 8, 16, 24]) 
    plt.xlabel('# of Cores')
    plt.ylabel('Bandwidth (MB/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('mem_bw_scaling_core.pdf')

