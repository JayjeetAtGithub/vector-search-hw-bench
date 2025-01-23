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
 * @param dis_metric The distance metric to use
 */
faiss::Index *CPU_create_hnsw_index(int64_t dim, int64_t ef, int64_t dis_metric) {
  // Use the default value of M in FAISS
  auto faiss_metric_type = (dis_metric == 0) ? faiss::MetricType::METRIC_L2
                                             : faiss::MetricType::METRIC_INNER_PRODUCT;
  auto index = new faiss::IndexHNSWFlat(dim, 32, faiss_metric_type);
  // Use the default value of efConstruction in FAISS
  index->hnsw.efConstruction = 40;
  index->hnsw.efSearch = ef;
  return index;
}

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string dataset_dir;
  app.add_option("-d,--dataset-dir", dataset_dir, "Path to the dataset");

  std::string gt_file;
  app.add_option("--gt-file", gt_file, "Path to the ground truth file");

  std::string index_file;
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

  int64_t dis_metric = 0;
  app.add_option("--metric", dis_metric, "Distance metric (0 = L2, 1 = IP)");

  int64_t skip_build = 0;
  app.add_option("--skip-build", skip_build, "Skip building the index");

  CLI11_PARSE(app, argc, argv);

  if (dataset_dir.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  if (!skip_build) {
    // Load the learn dataset
    std::string dataset_path_learn = dataset_dir + "/dataset.bin";
    int64_t dim_learn;
    auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &dim_learn, learn_limit);

    // Print information about the learn dataset
    std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << learn_limit
              << std::endl;
    preview_dataset(data_learn);

    // Set parameters
    int64_t n_list = int64_t(4 * std::sqrt(learn_limit));

    // Create the index
    faiss::Index *widx = CPU_create_hnsw_index(dim_learn, ef, dis_metric);

    // Add vectors to the index
    auto s = std::chrono::high_resolution_clock::now();
    widx->add(learn_limit, data_learn.data());
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Index: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    // Save the index to disk
    faiss::write_index(widx, index_file.c_str());
  }

  // Read the index from disk
  faiss::Index *ridx = faiss::read_index(index_file.c_str());

  // Load the search dataset
  std::string dataset_path_query = dataset_dir + "/query.bin";
  int64_t dim_query;
  auto data_query = read_bin_dataset(dataset_path_query.c_str(), &dim_query, search_limit);

  // Print information about the search dataset
  std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << search_limit
            << std::endl;
  preview_dataset(data_query);

  // Containers to hold the search results
  std::vector<faiss::idx_t> nns(top_k * search_limit);
  std::vector<float> dis(top_k * search_limit);

  // Perform the search
  auto s = std::chrono::high_resolution_clock::now();
  ridx->search(search_limit, data_query.data(), top_k, dis.data(), nns.data());
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  if (!gt_file.empty()) {
    std::vector<faiss::idx_t> gt_nns =
        read_vector(gt_file.c_str(), search_limit * top_k);

    // Calculate the recall
    int64_t recalls = 0;
    for (int64_t i = 0; i < search_limit; ++i) {
      for (int64_t n = 0; n < top_k; n++) {
        for (int64_t m = 0; m < top_k; m++) {
          if (nns[i * top_k + n] == gt_nns[i * top_k + m]) {
            recalls += 1;
          }
        }
      }
    }
    float recall = 1.0f * recalls / (top_k * search_limit);
    std::cout << "[INFO] Recall@" << top_k << ": " << recall << std::endl;
  }

  return 0;
}
