#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "CLI11.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#include "utils.h"

faiss::Index *CPU_create_hnsw_index(size_t dim) {
  auto idx = new faiss::IndexHNSWFlat(dim, 32);
  // idx->hnsw.efConstruction = 32;
  // idx->hnsw.efSearch = 16;
  return idx;
}

/**
 * @brief Create an IVF Flat index using the CPU
 * We use the default value of `n_probe` which is 1
 *
 * @param dim The dimension of the vectors
 * @param nlist The number of inverted lists
 */
faiss::Index *CPU_create_ivf_flat_index(size_t dim, size_t nlist) {
  auto quantizer = new faiss::IndexFlatL2(dim);
  auto index = new faiss::IndexIVFFlat(quantizer, dim, nlist, faiss::METRIC_L2);
  return index;
}

/**
 * @brief Create an IVF Flat index using the GPU
 * We use the default value of `n_probe` which is 1
 *
 * @param dim The dimension of the vectors
 * @param nlist The number of inverted lists
 */
faiss::Index *
GPU_create_ivf_flat_index(size_t dim, size_t nlist, std::string mem_type,
                          faiss::gpu::GpuResourcesProvider *provider,
                          int32_t cuda_device) {
  auto config = faiss::gpu::GpuIndexConfig();
  config.device = cuda_device;
  config.memorySpace = (mem_type == "cuda") ? faiss::gpu::MemorySpace::Device
                                            : faiss::gpu::MemorySpace::Unified;
  auto quantizer = new faiss::gpu::GpuIndexFlatL2(
      provider, dim, faiss::gpu::GpuIndexFlatConfig{config});
  auto index = new faiss::gpu::GpuIndexIVFFlat(
      provider, quantizer, dim, nlist, faiss::METRIC_L2,
      faiss::gpu::GpuIndexIVFFlatConfig{config});
  return index;
}

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string dataset;
  app.add_option("-d,--dataset", dataset, "Path to the dataset");

  bool persist_index = false;
  app.add_flag("--persist-index", persist_index, "Save the index");

  std::string mode = "cpu";
  app.add_option("--mode", mode, "Mode: cpu or gpu");

  std::string mem_type = "cuda";
  app.add_option("--mem-type", mem_type, "Memory type: cuda or managed");

  int32_t cuda_device = 0;
  app.add_option("--cuda-device", cuda_device, "The CUDA device to use");

  uint32_t learn_limit = 1000;
  app.add_option("--learn-limit", learn_limit,
                 "Limit the number of learn vectors");

  uint32_t search_limit = 1000;
  app.add_option("--search-limit", search_limit,
                 "Limit the number of search vectors");

  int32_t top_k = 10;
  app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

  CLI11_PARSE(app, argc, argv);

  if (dataset.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  // Preparing GPU resources
  auto provider = new faiss::gpu::StandardGpuResources();

  // Load the learn dataset
  uint32_t dim_learn, n_learn;
  float *data_learn;
  std::string dataset_path_learn = dataset + "/dataset.bin";
  read_dataset<float_t>(dataset_path_learn.c_str(), data_learn, &n_learn,
                        &dim_learn, learn_limit);

  // Print information about the learn dataset
  std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
            << std::endl;
  preview_dataset<float_t>(data_learn);

  // Set parameters
  size_t n_list = size_t(4 * std::sqrt(n_learn));

  // Create the index
  faiss::Index *idx;
  if (mode == "cpu") {
    idx = CPU_create_ivf_flat_index(dim_learn, n_list);
  } else {
    idx = GPU_create_ivf_flat_index(dim_learn, n_list, mem_type, provider,
                                    cuda_device);
  }

  // Train the index
  auto s = std::chrono::high_resolution_clock::now();
  idx->train(n_learn, data_learn);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Train: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Check if the index is trained
  if (!idx->is_trained) {
    std::cout << "[ERROR] Index is not trained" << std::endl;
    return 1;
  }

  // Add vectors to the index
  s = std::chrono::high_resolution_clock::now();
  idx->add(n_learn, data_learn);
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Index: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  if (persist_index) {
    // Save the index
    std::string index_path = "index.faiss";
    if (mode == "cpu") {
      faiss::write_index(idx, index_path.c_str());
    } else {
      faiss::write_index(faiss::gpu::index_gpu_to_cpu(idx), index_path.c_str());
    }
  }

  // Delete the learn dataset
  delete[] data_learn;

  // Load the search dataset
  uint32_t dim_query, n_query;
  float *data_query;
  std::string dataset_path_query = dataset + "/query.bin";
  read_dataset<float_t>(dataset_path_query.c_str(), data_query, &n_query,
                        &dim_query, search_limit);

  // Print information about the search dataset
  std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
            << std::endl;
  preview_dataset<float_t>(data_query);

  if (persist_index) {
    // Load the index
    faiss::Index *idx;
    if (mode == "cpu") {
      idx = faiss::read_index("index.faiss");
    } else {
      idx = faiss::gpu::index_cpu_to_gpu(provider, 0,
                                         faiss::read_index("index.faiss"));
    }
  }

  // Containers to hold the search results
  std::vector<faiss::idx_t> nns(top_k * n_query);
  std::vector<float> dis(top_k * n_query);

  // Perform the search
  s = std::chrono::high_resolution_clock::now();
  idx->search(n_query, data_query, top_k, dis.data(), nns.data());
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Delete the search dataset
  delete[] data_query;

  return 0;
}
