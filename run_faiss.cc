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
#include <faiss/gpu/GpuCloner.h>
#include <faiss/index_io.h>

#include "utils.h"

faiss::Index* CPU_create_ivf_flat_index(size_t dim, size_t nlist,
                                                    size_t nprobe) {
  return new faiss::IndexIVFFlat(
      new faiss::IndexFlatL2(dim), dim, nlist);
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <dataset> <mode> <limit> <top_k>" << std::endl;
    std::exit(1);
  }

  std::string dataset = argv[1];
  std::string mode = argv[2];
  uint32_t limit = std::atoi(argv[3]);
  int32_t top_k = std::atoi(argv[4]);

  // Print the arguments
  std::cout << "[ARG] dataset: " << dataset << std::endl;
  std::cout << "[ARG] mode: " << mode << std::endl;
  std::cout << "[ARG] limit: " << limit << std::endl;
  std::cout << "[ARG] top_k: " << top_k << std::endl;

  std::cout << std::endl;

  // Preparing GPU resources
  auto const n_gpus = faiss::gpu::getNumDevices();
  std::cout << "[INFO] Number of GPUs: " << n_gpus << std::endl;
  std::vector<faiss::gpu::GpuResourcesProvider*> res;
  std::vector<int> devs;
  for(int i = 0; i < n_gpus; i++) {
      res.push_back(new faiss::gpu::StandardGpuResources());
      devs.push_back(i);
  }

  // Load the learn dataset
  uint32_t dim_learn, n_learn;
  float *data_learn;
  std::string dataset_path_learn = dataset + "/base.bin";
  read_dataset2<float_t>(dataset_path_learn.c_str(), data_learn, &n_learn, &dim_learn, limit);
  
  // Print information about the learn dataset
  std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
            << std::endl;
  preview_dataset(data_learn);

  // Create and train the index
  auto idx = CPU_create_ivf_flat_index(dim_learn, 1024, 100);
  if (mode == "gpu_multiple") {
    idx = faiss::gpu::index_cpu_to_gpu_multiple(res, devs, idx);
  } else if (mode == "gpu_single") {
    idx = faiss::gpu::index_cpu_to_gpu(res[0], devs[0], idx);
  } else if (mode == "cpu") {
    // Do nothing
  } else {
    std::cerr << "[ERROR] Invalid mode" << std::endl;
    std::exit(1);
  }
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
  uint32_t dim_query, n_query;
  float *data_query;
  std::string dataset_path_query = dataset + "/query.bin";
  read_dataset2<float_t>(dataset_path_query.c_str(), data_query, &n_query, &dim_query, 10'000);
  
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
