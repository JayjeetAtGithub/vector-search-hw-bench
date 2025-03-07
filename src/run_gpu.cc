#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "CLI11.hpp"
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
  auto index = new faiss::gpu::GpuIndexFlatIP(
      provider, dim, faiss::gpu::GpuIndexFlatConfig{config});
  return index;
}

/**
 * @brief Create an IVF index using the GPU
 *
 * @param dim The dimension of the vectors
 * @param nlist The number of inverted lists
 * @param mem_type The memory type to use
 * @param provider The GPU resources provider
 * @param cuda_device The CUDA device to use
 */
faiss::Index *GPU_create_ivf_index(
    int64_t dim, int64_t nlist, std::string mem_type,
    faiss::gpu::GpuResourcesProvider *provider, int64_t cuda_device) {
  auto config = faiss::gpu::GpuIndexConfig();
  config.device = cuda_device;
  config.memorySpace = (mem_type == "cuda") ? faiss::gpu::MemorySpace::Device
                                            : faiss::gpu::MemorySpace::Unified;
  auto quantizer = new faiss::gpu::GpuIndexFlatIP(
      provider, dim, faiss::gpu::GpuIndexFlatConfig{config});
  auto index = new faiss::gpu::GpuIndexIVFFlat(
      provider, quantizer, dim, nlist, faiss::MetricType::METRIC_INNER_PRODUCT,
      faiss::gpu::GpuIndexIVFFlatConfig{config});
  return index;
}

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string index_type = "ivf";
  app.add_option("--index-type", index_type, "Type of index to use (ivf, flat)");

  std::string mem_type = "cuda";
  app.add_option("--mem-type", mem_type, "Memory type: cuda or managed");

  int64_t cuda_device = 0;
  app.add_option("--cuda-device", cuda_device, "The CUDA device to use");

  std::string calc_recall = "false";
  app.add_option("--calc-recall", calc_recall, "Calculate recall (true / false)");

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

  // This option is not used in the GPU version
  std::string dis_metric = "l2";
  app.add_option("--metric", dis_metric, "Distance metric to use (l2, ip)");

  int64_t skip_build = 0;
  app.add_option("--skip-build", skip_build, "Skip building the index");

  CLI11_PARSE(app, argc, argv);

  if (dataset_dir.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  // Preparing GPU resources
  auto provider = new faiss::gpu::StandardGpuResources();

  if (!skip_build) {
    // Load the learn dataset
    std::string dataset_path_learn = dataset_dir + "/dataset.bin";
    int64_t n_learn, dim_learn;
    auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);

    // Print information about the learn dataset
    std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
              << std::endl;
    preview_dataset(data_learn);

    // Set parameters
    int64_t n_list = int64_t(4 * std::sqrt(n_learn));

    // Create the index
    faiss::Index *widx_gpu;
    if (index_type == "ivf") {
      widx_gpu = GPU_create_ivf_index(dim_learn, n_list, mem_type, provider, cuda_device);
      widx_gpu->train(n_learn, data_learn.data());
    } else if (index_type == "flat") {
      widx_gpu = GPU_create_flat_index(dim_learn, mem_type, provider, cuda_device);
    } else {
      std::cerr << "[ERROR] Invalid index type" << std::endl;
      return 1;
    }

    // Add vectors to the index
    auto s = std::chrono::high_resolution_clock::now();
    widx_gpu->add(n_learn, data_learn.data());
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Index: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    // Save the index to disk
    auto widx_cpu = faiss::gpu::index_gpu_to_cpu(widx_gpu);
    faiss::write_index(widx_cpu, index_file.c_str());
  }

  if (skip_build) {
    // Read the index from disk
    faiss::Index *ridx_cpu = faiss::read_index(index_file.c_str());
    auto ridx_gpu = faiss::gpu::index_cpu_to_gpu(provider, cuda_device, ridx_cpu);

    if (index_type == "ivf") {
      dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(ridx_gpu)->nprobe = n_probe;
    }

    // Load the search dataset
    std::string dataset_path_query = dataset_dir + "/query.bin";
    int64_t n_query, dim_query;
    auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

    // Print information about the search dataset
    std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
              << std::endl;
    preview_dataset(data_query);

    // Containers to hold the search results
    std::vector<faiss::idx_t> nns(top_k * n_query);
    std::vector<float> dis(top_k * n_query);

    // Perform the search
    auto s = std::chrono::high_resolution_clock::now();
    for (int itr = 0; itr < 10; itr++) {
      ridx_gpu->search(n_query, data_query.data(), top_k, dis.data(), nns.data());
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Search: index: " << index_file.c_str() << " queries: " << n_query
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    if (calc_recall == "true") {
      std::string dataset_path_learn = dataset_dir + "/dataset.bin";
      int64_t n_learn, dim_learn;
      auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);

      faiss::Index *gt_idx_gpu = GPU_create_flat_index(dim_learn, mem_type, provider, cuda_device);
      gt_idx_gpu->add(n_learn, data_learn.data());
      std::vector<faiss::idx_t> gt_nns(top_k * n_query);
      std::vector<float> gt_dis(top_k * n_query);
      gt_idx_gpu->search(n_query, data_query.data(), top_k, gt_dis.data(), gt_nns.data());
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
    }
  }

  return 0;
}
