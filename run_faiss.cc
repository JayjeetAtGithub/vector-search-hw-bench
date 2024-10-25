#include <chrono>
#include <cmath>
#include <filesystem>
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

#include "cuda_profiler_api.h"
#include "utils.h"

/**
 * @brief Create a HNSW index using the CPU
 *
 * @param dim The dimension of the vectors
 * @param ef The number of neighbors to explore
 */
faiss::Index *CPU_create_hnsw_index(int64_t dim, int64_t ef) {
  // Use the default value of M in FAISS
  auto index = new faiss::IndexHNSWFlat(dim, 32);
  // Use the default value of efConstruction in FAISS
  index->hnsw.efConstruction = 40;
  index->hnsw.efSearch = ef;
  return index;
}

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

  std::string dataset_dir;
  app.add_option("-d,--dataset-dir", dataset_dir, "Path to the dataset");

  std::string gt_file;
  app.add_option("--gt-file", gt_file, "Path to the ground truth file");

  std::string mode = "cpu";
  app.add_option("--mode", mode, "Mode: cpu or gpu");

  std::string algo = "ivf";
  app.add_option("--algo", algo, "Algorithm to run: hnsw or ivf");

  std::string mem_type = "cuda";
  app.add_option("--mem-type", mem_type, "Memory type: cuda or managed");

  int64_t cuda_device = 0;
  app.add_option("--cuda-device", cuda_device, "The CUDA device to use");

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

  int64_t ef = 256;
  app.add_option("--ef", ef, "Number of neighbors to explore");

  CLI11_PARSE(app, argc, argv);

  if (dataset_dir.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  // Preparing GPU resources
  auto provider = new faiss::gpu::StandardGpuResources();

  // Load the learn dataset
  std::string dataset_path_learn = dataset_dir + "/dataset.bin";
  int64_t dim_learn, n_learn;
  auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn,
                                     &dim_learn, learn_limit);

  // Print information about the learn dataset
  std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
            << std::endl;
  preview_dataset(data_learn);

  // Set parameters
  int64_t n_list = int64_t(4 * std::sqrt(n_learn));
  std::string id = algo + " / " + mode + " / " + mem_type;

  // Create the index (always on the GPU)
  faiss::Index *idx;
  if (algo == "hnsw") {
    idx = CPU_create_hnsw_index(dim_learn, ef);
  } else if (algo == "ivf") {
    idx = GPU_create_ivf_flat_index(dim_learn, n_list, n_probe, mem_type,
                                    provider, cuda_device);
  } else {
    std::cout << "[ERROR] Invalid algorithm" << std::endl;
    return 1;
  }

  if (algo == "ivf") {
    // Train the index
    auto s = std::chrono::high_resolution_clock::now();
    idx->train(n_learn, data_learn.data());
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Train [" << id << "]: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    // Check if the index is trained
    if (!idx->is_trained) {
      std::cout << "[ERROR] Index is not trained" << std::endl;
      return 1;
    }
  }

  // Add vectors to the index
  auto s = std::chrono::high_resolution_clock::now();
  idx->add(n_learn, data_learn.data());
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Index [" << id << "]: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // If mode is CPU, copy the index to CPU
  if (mode == "cpu" && algo == "ivf") {
    idx = faiss::gpu::index_gpu_to_cpu(idx);
  }

  // Load the search dataset
  std::string dataset_path_query = dataset_dir + "/query.bin";
  int64_t dim_query, n_query;
  auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query,
                                     &dim_query, search_limit);

  // Print information about the search dataset
  std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
            << std::endl;
  preview_dataset(data_query);

  // Containers to hold the search results
  std::vector<faiss::idx_t> nns(top_k * n_query);
  std::vector<float> dis(top_k * n_query);

  cudaProfilerStart();
  // Perform the search
  s = std::chrono::high_resolution_clock::now();
  idx->search(n_query, data_query.data(), top_k, dis.data(), nns.data());
  e = std::chrono::high_resolution_clock::now();
  cudaProfilerStop();
  std::cout
      << "[TIME] Search [" << id << "]: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Run bruteforce experiments
  std::vector<faiss::idx_t> gt_nns(top_k * n_query);
  std::vector<float> gt_dis(top_k * n_query);

  if (!gt_file.empty()) {
    if (std::filesystem::exists(gt_file)) {
      std::cout << "[INFO] Reading ground truth from file" << std::endl;
      gt_nns = read_vector(gt_file.c_str(), top_k * n_query);
    } else {
      std::cout
          << "[INFO] Ground truth file not found. Calculating ground truth"
          << std::endl;
      auto brute_force_index =
          GPU_create_flat_index(dim_learn, mem_type, provider, cuda_device);
      brute_force_index->add(n_learn, data_learn.data());
      brute_force_index->search(n_query, data_query.data(), top_k,
                                gt_dis.data(), gt_nns.data());
      std::cout << "[INFO] Writing ground truth to file" << std::endl;
      write_vector(gt_file.c_str(), gt_nns.data(), top_k * n_query);
    }
  }

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

  return 0;
}
