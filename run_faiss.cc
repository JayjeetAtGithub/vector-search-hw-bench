#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#include "utils.h"

std::unique_ptr<faiss::Index> CPU_create_ivf_flat_index(size_t dim, size_t nlist,
                                                    size_t nprobe) {
  auto index = std::make_unique<faiss::IndexIVFFlat>(
      new faiss::IndexFlatL2(dim), dim, nlist);
  return index;
}

std::unique_ptr<faiss::Index>
GPU_create_ivf_flat_gpu_index(size_t dim, size_t nlist, size_t nprobe) {
  auto res = new faiss::gpu::StandardGpuResources();
  auto index = std::make_unique<faiss::gpu::GpuIndexIVFFlat>(
      res, new faiss::gpu::GpuIndexFlatL2(res, dim), dim, nlist);
  return index;
}

std::unique_ptr<faiss::Index> create_ivf_flat_index(size_t dim, size_t nlist,
                                                size_t nprobe, std::string mode) {
  if (mode == "cpu") {
    return CPU_create_ivf_flat_index(dim, nlist, nprobe);
  } else if (mode == "gpu") {
    return GPU_create_ivf_flat_gpu_index(dim, nlist, nprobe);
  } else {
    std::cerr << "[ERROR] Invalid mode: " << mode << std::endl;
    std::exit(1);
  }
}

int main(int argc, char **argv) {
  std::string dataset = argv[1];
  std::string mode = argv[2];

  int32_t top_k = 100;

  std::cout << "[ARG] dataset: " << dataset << std::endl;
  std::cout << "[ARG] top_k: " << top_k << std::endl;
  std::cout << "[ARG] mode: " << mode << std::endl;

  std::cout << std::endl;

  // Load the learn dataset
  size_t dim_learn, n_learn;
  float *data_learn;
  std::string dataset_path_learn = dataset + "/base.fvecs";
  read_dataset(dataset_path_learn.c_str(), data_learn, &dim_learn, &n_learn);
  n_learn = 1'000'000;
  
  // Print information about the learn dataset
  std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
            << std::endl;
  preview_dataset(data_learn);

  // Create and train the index
  auto const idx = create_ivf_flat_index(dim_learn, 1024, 100, mode);
  idx->train(n_learn, data_learn);

  // Add vectors to the index
  auto s = std::chrono::high_resolution_clock::now();
  idx->add(n_learn, data_learn);
  auto e = std::chrono::high_resolution_clock::now();
  std::cout
      << "[TIME] Index: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
      << " ms" << std::endl;

  // Delete the learn dataset
  delete[] data_learn;

  std::cout << std::endl;

  // Load the query dataset
  size_t dim_query, n_query;
  float *data_query;
  std::string dataset_path_query = dataset + "/learn.fvecs";
  read_dataset(dataset_path_query.c_str(), data_query, &dim_query, &n_query);
  n_query = 10'000;
  
  // Print information about the query dataset
  std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
            << std::endl;
  preview_dataset(data_query);

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

  // Delete the query dataset
  delete[] data_query;

  return 0;
}
