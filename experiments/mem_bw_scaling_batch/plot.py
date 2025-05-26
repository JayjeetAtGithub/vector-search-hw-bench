import os
import re
import sys
import pandas as pd

def find_pcm_logs(root_dir='.'):
    pcm_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pcm_log'):
                pcm_files.append(os.path.join(dirpath, filename))
    return pcm_files

def compute_median_bandwidth(file_path):
    df = pd.read_csv(file_path)
    return df['Read'].max(), df['Write'].max(), df['Memory'].max()

def get_batch_size(filename):
    match = re.search(r'_(\d+)s\.pcm_log_fixed$', filename)
    if match:
        batch_size = int(match.group(1))
        return batch_size
    else:
        print("Batch size not found in filename.")
        sys.exit(1)

if __name__ == '__main__':
    pcm_logs = find_pcm_logs()
    print(f"Found {len(pcm_logs)} .pcm_log files.")
    data = list()

    for log_file in pcm_logs:
        # Remove the first line of the log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        with open(f"{log_file}_fixed", 'w') as f:
            f.writelines(lines[1:])

        batch_size = get_batch_size(f"{log_file}_fixed")
        rd, wr, total = compute_median_bandwidth(f"{log_file}_fixed")
        print(f"Batch Size: {batch_size}, Read: {rd:.2f} GiB/s, Write: {wr:.2f} GiB/s, Total: {total:.2f} GiB/s")
        data.append(
            {
                "bs": batch_size,
                "rd": rd,
                "wr": wr,
                "total": total,
            })

    df = pd.DataFrame(data)
    print(df)
