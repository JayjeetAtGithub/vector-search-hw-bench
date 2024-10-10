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

/**
 * @brief Create a Flat index using the GPU
 *
 * @param dim The dimension of the vectors
 * @param mem_type The memory type to use
 * @param provider The GPU resources provider
 * @param cuda_device The CUDA device to use
 */
faiss::Index *GPU_create_flat_index(int64_t dim, std::string mem_type,
                                    faiss::gpu::GpuResourcesProvider *provider,
                                    int64_t cuda_device) {
  auto config = faiss::gpu::GpuIndexConfig();
  config.device = cuda_device;
  config.memorySpace = (mem_type == "cuda") ? faiss::gpu::MemorySpace::Device
                                            : faiss::gpu::MemorySpace::Unified;
  auto index = new faiss::gpu::GpuIndexFlatL2(
      provider, dim, faiss::gpu::GpuIndexFlatConfig{config});
  return index;
}

/**
 * @brief Create an IVF Flat index using the GPU
 *
 * @param dim The dimension of the vectors
 * @param nlist The number of inverted lists
 * @param mem_type The memory type to use
 * @param provider The GPU resources provider
 * @param cuda_device The CUDA device to use
 */
faiss::Index *GPU_create_ivf_flat_index(
    int64_t dim, int64_t nlist, int64_t nprobe, std::string mem_type,
    faiss::gpu::GpuResourcesProvider *provider, int64_t cuda_device) {
  auto config = faiss::gpu::GpuIndexConfig();
  config.device = cuda_device;
  config.memorySpace = (mem_type == "cuda") ? faiss::gpu::MemorySpace::Device
                                            : faiss::gpu::MemorySpace::Unified;
  auto quantizer = new faiss::gpu::GpuIndexFlatL2(
      provider, dim, faiss::gpu::GpuIndexFlatConfig{config});
  auto index = new faiss::gpu::GpuIndexIVFFlat(
      provider, quantizer, dim, nlist, faiss::METRIC_L2,
      faiss::gpu::GpuIndexIVFFlatConfig{config});
  index->nprobe = nprobe;
  return index;
}

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string dataset;
  app.add_option("-d,--dataset", dataset, "Path to the dataset");

  std::string mode = "cpu";
  app.add_option("--mode", mode, "Mode: cpu or gpu");

  std::string mem_type = "cuda";
  app.add_option("--mem-type", mem_type, "Memory type: cuda or managed");

  int64_t cuda_device = 0;
  app.add_option("--cuda-device", cuda_device, "The CUDA device to use");

  int64_t learn_limit = 1000;
  app.add_option("--learn-limit", learn_limit,
                 "Limit the number of learn vectors");

  int64_t search_limit = 1000;
  app.add_option("--search-limit", search_limit,
                 "Limit the number of search vectors");

  int64_t top_k = 10;
  app.add_option("-k,--top-k", top_k, "Number of nearest neighbors");

  int64_t n_probe = 32;
  app.add_option("--n-probe", n_probe, "Number of probes");

  CLI11_PARSE(app, argc, argv);

  if (dataset.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  // Preparing GPU resources
  auto provider = new faiss::gpu::StandardGpuResources();

  // Load the learn dataset
  int64_t dim_learn, n_learn;
  float *data_learn;
  std::string dataset_path_learn = dataset + "/dataset.bin";
  read_dataset<float_t>(dataset_path_learn.c_str(), data_learn, &n_learn,
                        &dim_learn, learn_limit);

  // Print information about the learn dataset
  std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
            << std::endl;
  preview_dataset<float_t>(data_learn);

  // Set parameters
  int64_t n_list = int64_t(4 * std::sqrt(n_learn));

  // Create the index (always on the GPU)
  faiss::Index *idx = GPU_create_ivf_flat_index(
      dim_learn, n_list, n_probe, mem_type, provider, cuda_device);

  // Train the index
  auto s = std::chrono::high_resolution_clock::now();
  idx->train(n_learn, data_learn);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Train [gpu]: "
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
      << "[TIME] Index [gpu]: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // If mode is CPU, copy the index to CPU
  if (mode == "cpu") {
    idx = faiss::gpu::index_gpu_to_cpu(idx);
  }

  // Load the search dataset
  int64_t dim_query, n_query;
  float *data_query;
  std::string dataset_path_query = dataset + "/query.bin";
  read_dataset<float_t>(dataset_path_query.c_str(), data_query, &n_query,
                        &dim_query, search_limit);

  // Print information about the search dataset
  std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
            << std::endl;
  preview_dataset<float_t>(data_query);

  // Containers to hold the search results
  std::vector<faiss::idx_t> nns(top_k * n_query);
  std::vector<float> dis(top_k * n_query);

  // Perform the search
  s = std::chrono::high_resolution_clock::now();
  idx->search(n_query, data_query, top_k, dis.data(), nns.data());
  e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Search [" << mode << "]: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Run bruteforce experiments
  std::vector<faiss::idx_t> gt_nns(top_k * n_query);
  std::vector<float> gt_dis(top_k * n_query);
  auto brute_force_index =
      GPU_create_flat_index(dim_learn, mem_type, provider, cuda_device);
  brute_force_index->add(n_learn, data_learn);
  brute_force_index->search(n_query, data_query, top_k, gt_dis.data(),
                            gt_nns.data());

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

  // Delete the datasets
  delete[] data_learn;
  delete[] data_query;

  return 0;
}
