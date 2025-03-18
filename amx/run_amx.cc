#include "bf.hpp"
#include "utils.h"

int main() {
    int64_t learn_limit = 100000;
    int64_t search_limit = 1000;
    int32_t dim = 200;
    int32_t top_k = 10;

    std::string dataset_path_learn = "/workspace/dataset/t2i/dataset.bin";
    int64_t n_learn, dim_learn;
    auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);
    
    std::string dataset_path_query =  "/workspace/dataset/t2i/query.bin";
    int64_t n_query, dim_query;
    auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

    uint64_t total_flop = (n_learn * n_query) * (2 * dim - 1);
    std::cout << "Total FLOP: " << total_flop << std::endl;

    auto bf_search = std::make_shared<BruteForceSearch>(dim, n_query, n_learn);

    for (int i = 0; i < 20; i++) {
        auto s = std::chrono::high_resolution_clock::now();
        auto results = bf_search->search_ip_amx(data_query, data_learn, top_k);
        auto e = std::chrono::high_resolution_clock::now();
        std::cout
            << "[TIME] Search: [ index: " << "flat" << " ][ # queries: " << n_query << " ]: "
            << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count()
            << " us" << std::endl;

        std::cout << "Previewing search results" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << "Query " << i << std::endl;
            for (int j = 0; j < top_k; j++) {
                std::cout << results[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
