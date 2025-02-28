#include "bf.hpp"
#include "utils.h"

int main() {
    int64_t learn_limit = 1000000;
    int64_t search_limit = 10000;
    int32_t dim = 200;

    std::string dataset_path_learn = "/workspace/dataset/deep1b/dataset.bin";
    int64_t n_learn, dim_learn;
    auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);
    
    std::string dataset_path_query =  "/workspace/dataset/deep1b/query.bin";
    int64_t n_query, dim_query;
    auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

    auto bf_search = std::make_shared<BruteForceSearch>(dim);
    bf_search->search_ip_amx(data_query, n_query, data_learn, n_learn, 10);
    return 0;
}
