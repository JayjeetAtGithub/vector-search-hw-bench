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

/**
 * @brief Create a HNSW index using the CPU
 *
 * @param dim The dimension of the vectors
 * @param ef The number of neighbors to explore
 */
faiss::Index *CPU_create_hnsw_index(int64_t dim, int64_t ef) {
  auto index = new faiss::IndexHNSWFlat(dim, 32);
  index->hnsw.efConstruction = ef;
  index->hnsw.efSearch = ef;
  return index;
}

/**
 * @brief Create a Flat index using the CPU
 *
 * @param dim The dimension of the vectors
 */
faiss::Index *CPU_create_flat_index(int64_t dim) {
  auto index = new faiss::IndexFlatL2(dim);
  return index;
}

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string dataset;
  app.add_option("-d,--dataset", dataset, "Path to the dataset");

  std::string mode = "train";
  app.add_option("--mode", mode, "Whether to train an index or search an index");

  std::string hnsw_index_path;
  app.add_option("--hnsw-index", hnsw_index_path, "Path to a hnsw index on disk");

  std::string bf_index_path;
  app.add_option("--bf-index", bf_index_path, "Path to a brute-force index on disk");

  int64_t learn_limit = 1000;
  app.add_option("--learn-limit", learn_limit,
                 "Limit the number of learn vectors");

  int64_t search_limit = 1000;
  app.add_option("--search-limit", search_limit,
                 "Limit the number of search vectors");

  int64_t top_k = 10;
  app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

  int64_t ef = 256;
  app.add_option("--ef", ef, "Number of neighbors to explore");

  CLI11_PARSE(app, argc, argv);

  if (mode == "train") {
    int64_t n_learn, dim_learn;
    float *data_learn;

    // Load the learn dataset
    std::string dataset_path_learn = dataset + "/dataset.bin";
    read_dataset<float_t>(dataset_path_learn.c_str(), data_learn, &n_learn,
                          &dim_learn, learn_limit);

    // Print information about the learn dataset
    std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
              << std::endl;
    preview_dataset<float_t>(data_learn);

    // Create the indexes
    auto hnsw_idx = CPU_create_hnsw_index(dim_learn, ef);
    auto bf_idx = CPU_create_flat_index(dim_learn);

    // Add vectors to the indexes
    auto s = std::chrono::high_resolution_clock::now();
    hnsw_idx->add(n_learn, data_learn);
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
      << "[TIME] Index: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

    bf_idx->add(n_learn, data_learn);

    // Write out the indexes to files
    faiss::write_index(hnsw_idx, hnsw_index_path.c_str());
    faiss::write_index(bf_idx, bf_index_path.c_str());

    delete []data_learn;
  } else {
    // Load the hnsw and brute force indexes
    auto hnsw_idx = faiss::read_index(hnsw_index_path.c_str());
    auto bf_idx = faiss::read_index(bf_index_path.c_str());

    // Load the search dataset
    int64_t n_query, dim_query;
    float *data_query;
    std::string dataset_path_query = dataset + "/query.bin";
    read_dataset<float_t>(dataset_path_query.c_str(), data_query, &n_query,
                          &dim_query, search_limit);

    // Print information about the search dataset
    std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
              << std::endl;
    preview_dataset<float_t>(data_query);

    // Perform the HNSW search
    std::vector<faiss::idx_t> nns(top_k * n_query);
    std::vector<float> dis(top_k * n_query);
    auto s = std::chrono::high_resolution_clock::now();
    hnsw_idx->search(n_query, data_query, top_k, dis.data(), nns.data());
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Search: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    // Perform a brute-force search
    std::vector<faiss::idx_t> gt_nns(top_k * n_query);
    std::vector<float> gt_dis(top_k * n_query);
    bf_idx->search(n_query, data_query, top_k, gt_dis.data(), gt_nns.data());

    // Calculate the recall
    int64_t recalls = 0;
    for (int64_t i = 0; i < n_query; ++i) {
      for (int64_t n = 0; n < top_k; n++) {
        for (int64_t m = 0; m < top_k; m++) {
          if (nns[i * top_k + n] == gt_nns[i * top_k + m]) {
            recalls += 1;
          }
        }
      }
    }
    float recall = 1.0f * recalls / (top_k * n_query);
    std::cout << "[INFO] Recall@" << top_k << ": " << recall << std::endl;
    delete[] data_query;
  }

  return 0;
}
