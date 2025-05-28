#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "CLI11.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include "utils.h"

#include <sstream>
#include <iostream>
#include <functional>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

struct System
{
    static void stat(const std::string& name,std::function<void()> body) {
        std::string filename = name.find(".perf-stat") == std::string::npos ? (name + ".perf-stat") : name;

        // Launch profiler
        pid_t pid;
        std::stringstream s;
        s << getpid();
        pid = fork();
        if (pid == 0) {
            auto fd=open("/dev/null",O_RDWR);
            dup2(fd,1);
            dup2(fd,2);
            exit(execl(
                "/usr/bin/perf",
                "perf",
                "stat",
                "-a",

                "-e", "l1d_cache",
                "-e", "l1d_cache_rd",
                "-e", "l1d_cache_refill",
                "-e", "l2d_cache_rd",
                "-e", "l2d_cache_refill",
                "-e", "ll_cache_rd",
                "-e", "ll_cache_miss_rd",
                "-e", "LLC-loads",
                "-e", "LLC-load-misses",
                "-e", "L1-dcache-loads",
                "-e", "L1-dcache-load-misses",
                
                "-o",
                filename.c_str(),
                nullptr
            ));
        }

        // Run body
        body();

        // Kill profiler  
        kill(pid,SIGINT);
        waitpid(pid,nullptr,0);
    }
};

int main(int argc, char **argv) {
    CLI::App app{"Run FAISS Benchmarks"};
    argv = app.ensure_utf8(argv);

    std::string index_type = "hnsw";
    app.add_option("--index-type", index_type, "Type of index to use (hnsw, ivf, flat)");

    std::string dataset_dir;
    app.add_option("-d,--dataset-dir", dataset_dir, "Path to the dataset");

    std::string index_file = "index.faiss";
    app.add_option("--index-file", index_file, "Path to the index file");

    int64_t learn_limit = 10000;
    app.add_option("--learn-limit", learn_limit,
                    "Limit the number of learn vectors");

    int64_t search_limit = 10000;
    app.add_option("--search-limit", search_limit,
                    "Limit the number of search vectors");

    int64_t top_k = 10;
    app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

    int64_t ef = 256;
    app.add_option("--ef", ef, "Number of neighbors to explore");

    int64_t n_probe = 32;
    app.add_option("--n-probe", n_probe, "Number of probes");

    int64_t M = 32;
    app.add_option("--M", M, "Number of neighbors to connect");

    std::string dis_metric = "l2";
    app.add_option("--metric", dis_metric, "Distance metric to use (l2, ip)");

    std::string out;
    app.add_option("--out", out, "Path to the output file");

    CLI11_PARSE(app, argc, argv);

    if (dataset_dir.empty()) {
        std::cerr << "[ERROR] Please provide a dataset" << std::endl;
        return 1;
    }
   
    // Read the index from disk
    faiss::Index *ridx = faiss::read_index(index_file.c_str());
    if (index_type == "ivf") {
        dynamic_cast<faiss::IndexIVFFlat*>(ridx)->nprobe = n_probe;
    } else if (index_type == "hnsw") {
        dynamic_cast<faiss::IndexHNSWFlat*>(ridx)->hnsw.efSearch = ef;
    }

    // Load the search dataset
    std::string dataset_path_query = dataset_dir + "/query.bin";
    int64_t n_query, dim_query;
    auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

    // Print information about the search dataset
    std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
                << std::endl;
    preview_dataset(data_query);

    size_t qif = 5;
    std::vector<faiss::idx_t> nns(top_k * n_query * pow(2, qif));
    std::vector<float> dis(top_k * n_query * pow(2, qif));
    std::cout << "Using " << n_query * pow(2, qif) << " queries" << std::endl;

    for (int i = 0; i < qif; i++) {
        data_query.insert(data_query.end(), data_query.begin(), data_query.end());
    }

    // Perform the search
    System::stat(out, [&]() {
        ridx->search(n_query * pow(2, qif), data_query.data(), top_k, dis.data(), nns.data());
    });
    
    delete ridx;
    return 0;
}
