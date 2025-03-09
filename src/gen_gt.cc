#include <cmath>
#include <iostream>
#include <vector>

#include "CLI11.hpp"
#include <faiss/gpu/GpuIndexFlat.h>
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

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string mem_type = "cuda";
  app.add_option("--mem-type", mem_type, "Memory type: cuda or managed");

  int64_t cuda_device = 0;
  app.add_option("--cuda-device", cuda_device, "The CUDA device to use");

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

  CLI11_PARSE(app, argc, argv);

  if (dataset_dir.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  auto provider = new faiss::gpu::StandardGpuResources();

  std::string dataset_path_learn = dataset_dir + "/dataset.bin";
  int64_t n_learn, dim_learn;
  auto data_learn = read_bin_dataset(dataset_path_learn.c_str(), &n_learn, &dim_learn, learn_limit);

  std::string dataset_path_query = dataset_dir + "/query.bin";
  int64_t n_query, dim_query;
  auto data_query = read_bin_dataset(dataset_path_query.c_str(), &n_query, &dim_query, search_limit);

  faiss::Index *gt_idx_gpu = GPU_create_flat_index(dim_learn, mem_type, provider, cuda_device);
  gt_idx_gpu->add(n_learn, data_learn.data());
      
  std::vector<faiss::idx_t> gt_nns(top_k * n_query);
  std::vector<float> gt_dis(top_k * n_query);
      
  gt_idx_gpu->search(n_query, data_query.data(), top_k, gt_dis.data(), gt_nns.data());

  return 0;
}
