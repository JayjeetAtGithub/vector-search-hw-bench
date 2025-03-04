#include "bf.hpp"
#include "utils.h"

int main() {
    int64_t learn_limit = 1000000;
    int64_t search_limit = 10000;
    int32_t dim = 200;
    int32_t top_k = 10;

    std::string dataset_path_learn = "/workspace/dataset/deep1b/dataset.bin";
    int64_t n_learn, dim_learn;
    auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);
    
    std::string dataset_path_query =  "/workspace/dataset/deep1b/query.bin";
    int64_t n_query, dim_query;
    auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

    auto bf_search = std::make_shared<BruteForceSearch>(dim);
    auto results = bf_search->search_ip_amx(data_query, n_query, data_learn, n_learn, top_k);
    std::cout << "Results preview: " << std::endl;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < top_k; j++) {
            std::cout << results[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
