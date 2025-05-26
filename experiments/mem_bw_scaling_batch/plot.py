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
    df = df.iloc[1:]
    return (float(df['System.4'].max().strip()), 
            float(df['System.5'].max().strip()), 
            float(df['System.6'].max().strip()))

def get_batch_size(filename):
    match = re.search(r'_(\d+)s\.pcm_log$', filename)
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
        batch_size = get_batch_size(log_file)
        rd, wr, total = compute_median_bandwidth(log_file)
        data.append(
            {
                "bs": batch_size,
                "rd": rd,
                "wr": wr,
                "total": total,
            })

    df = pd.DataFrame(data)
    print(df)
