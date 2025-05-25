import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv("perf_data.csv")

    df['read_bytes'] = df['read_mib'] * (1024 * 1024)
    df['write_bytes'] = df['write_mib'] * (1024 * 1024)

    df['tota_read_bytes_per_iteration'] = df['read_bytes'] / 20
    df['total_write_bytes_per_iteration'] = df['write_bytes'] / 20

    df['avg_read_bytes_per_query'] = df['tota_read_bytes_per_iteration'] / 10000
    df['avg_write_bytes_per_query'] = df['total_write_bytes_per_iteration'] / 10000

    df['avg_read_mib_per_query'] = df['avg_read_bytes_per_query'] / (1024 * 1024)
    df['avg_write_mib_per_query'] = df['avg_write_bytes_per_query'] / (1024 * 1024)
    df['avg_total_mib_per_query'] = df['avg_read_mib_per_query'] + df['avg_write_mib_per_query']
    print(df)

    # Plot dataset size vs avg mib per query
    plt.figure(figsize=(5, 3))
    plt.plot(df['dataset_size'], df['avg_read_mib_per_query'], marker='o', linestyle='-', color='b')
    plt.plot(df['dataset_size'], df['avg_write_mib_per_query'], marker='^', linestyle='-', color='r')
    plt.plot(df['dataset_size'], df['avg_total_mib_per_query'], marker='X', linestyle='-', color='g')
    plt.xlabel('Dataset Size')
    plt.ylabel('MiB')
    plt.grid()
    plt.legend(['Avg. Read MiB / Query', 'Avg. Write MiB / Query', 'Avg. Total MiB / Query'])
    plt.tight_layout(pad=1.0)
    plt.savefig('data_read.pdf')
