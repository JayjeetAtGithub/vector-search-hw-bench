#include "bf.hpp"
#include "utils.h"
#include "CLI11.hpp"

int main() {
    CLI::App app{"Run FAISS Benchmarks"};
    argv = app.ensure_utf8(argv);
  
    std::string index_type = "ivf";
    app.add_option("--index-type", index_type, "Type of index to use (ivf, flat)");

    std::string calc_recall = "false";
    app.add_option("--calc-recall", calc_recall, "Calculate recall (true / false)");
  
    std::string dataset_dir;
    app.add_option("-d,--dataset-dir", dataset_dir, "Path to the dataset");

    int64_t learn_limit = 10000;
    app.add_option("--learn-limit", learn_limit,
                   "Limit the number of learn vectors");
  
    int64_t search_limit = 10000;
    app.add_option("--search-limit", search_limit,
                   "Limit the number of search vectors");
  
    int64_t top_k = 10;
    app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");
  
    int64_t n_probe = 32;
    app.add_option("--n-probe", n_probe, "Number of probes");

    CLI11_PARSE(app, argc, argv);
  
    if (dataset_dir.empty()) {
      std::cerr << "[ERROR] Please provide a dataset" << std::endl;
      return 1;
    }

    std::string dataset_path_learn = "/workspace/dataset/t2i/dataset.bin";
    int64_t n_learn, dim_learn;
    auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);
    
    std::string dataset_path_query =  "/workspace/dataset/t2i/query.bin";
    int64_t n_query, dim_query;
    auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

    uint64_t total_flop = (n_learn * n_query) * (2 * dim - 1);
    std::cout << "Total FLOP: " << total_flop << std::endl;

    auto bf_search = std::make_shared<BruteForceSearch>(dim, n_query, n_learn);

    for (int i = 0; i < 10; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        auto results = bf_search->search_ip_amx(data_query, data_learn, top_k);
        auto e = std::chrono::high_resolution_clock::now();
        std::cout
            << "[TIME] Search: [ index: " << "flat" << " ][ # queries: " << n_query << " ]: "
            << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count()
            << " us" << std::endl;
    }

    return 0;
}
