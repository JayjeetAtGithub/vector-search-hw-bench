#include <chrono>
#include <iostream>
#include <vector>

#include "CLI11.hpp"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

#include "utils.h"

faiss::Index *CPU_create_ivf_flat_index(size_t dim, size_t nlist,
                                        size_t nprobe) {
  auto quantizer = new faiss::IndexFlatL2(dim);
  auto index = new faiss::IndexIVFFlat(quantizer, dim, nlist, faiss::METRIC_L2);
  index->nprobe = nprobe;
  return index;
}

faiss::Index *GPU_create_ivf_flat_index(size_t dim, size_t nlist, size_t nprobe,
                                        std::string mem_type,
                                        int32_t cuda_device) {
  auto res = new faiss::gpu::StandardGpuResources();
  auto config = faiss::gpu::GpuIndexConfig();
  config.device = cuda_device;
  config.memorySpace = (mem_type == "cuda") ? faiss::gpu::MemorySpace::Device
                                            : faiss::gpu::MemorySpace::Unified;
  auto quantizer = new faiss::gpu::GpuIndexFlatL2(
      res, dim, faiss::gpu::GpuIndexFlatConfig{config});
  auto index = new faiss::gpu::GpuIndexIVFFlat(
      res, quantizer, dim, nlist, faiss::METRIC_L2,
      faiss::gpu::GpuIndexIVFFlatConfig{config});
  index->nprobe = nprobe;
  return index;
}

int main(int argc, char **argv) {
  CLI::App app{"Run FAISS Benchmarks"};
  argv = app.ensure_utf8(argv);

  std::string op;
  app.add_option("--op", op, "Operation to perform: index or search");

  std::string dataset;
  app.add_option("-d,--dataset", dataset, "Path to the dataset");

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

  int32_t n_list = 100;
  app.add_option("-n,--n-list", n_list, "Number of inverted lists");

  int32_t n_probe = 10;
  app.add_option("-p,--n-probe", n_probe, "Number of cells to probe");

  CLI11_PARSE(app, argc, argv);

  if (op.empty()) {
    std::cerr << "[ERROR] Please provide an operation" << std::endl;
    return 1;
  }
  if (dataset.empty()) {
    std::cerr << "[ERROR] Please provide a dataset" << std::endl;
    return 1;
  }

  // Preparing GPU resources
  auto const n_gpus = faiss::gpu::getNumDevices();
  std::cout << "[INFO] Number of GPUs: " << n_gpus << std::endl;
  std::vector<faiss::gpu::GpuResourcesProvider *> res;
  std::vector<int> devs;
  for (int i = 0; i < n_gpus; i++) {
    res.push_back(new faiss::gpu::StandardGpuResources());
    devs.push_back(i);
  }

  if (op == "index") {
    // Load the learn dataset
    uint32_t dim_learn, n_learn;
    float *data_learn;
    std::string dataset_path_learn = dataset + "/base.1B.bin";
    read_dataset<float_t>(dataset_path_learn.c_str(), data_learn, &n_learn,
                          &dim_learn, learn_limit);

    // Print information about the learn dataset
    std::cout << "[INFO] Learn dataset shape: " << dim_learn << " x " << n_learn
              << std::endl;
    preview_dataset(data_learn);

    // Create the index
    faiss::Index *idx;
    if (mode == "cpu") {
      idx = CPU_create_ivf_flat_index(dim_learn, n_list, n_probe);
    } else {
      idx = GPU_create_ivf_flat_index(dim_learn, n_list, n_probe, mem_type,
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

    // Save the index
    std::string index_path = "index.faiss";
    if (mode == "cpu") {
      faiss::write_index(idx, index_path.c_str());
    } else {
      faiss::write_index(faiss::gpu::index_gpu_to_cpu(idx), index_path.c_str());
    }

    // Delete the learn dataset
    delete[] data_learn;
  }

  if (op == "search") {
    // Load the search dataset
    uint32_t dim_query, n_query;
    float *data_query;
    std::string dataset_path_query = dataset + "/query.bin";
    read_dataset<float_t>(dataset_path_query.c_str(), data_query, &n_query,
                          &dim_query, search_limit);

    // Print information about the search dataset
    std::cout << "[INFO] Query dataset shape: " << dim_query << " x " << n_query
              << std::endl;
    preview_dataset(data_query);

    // Load the index
    faiss::Index *idx;
    if (mode == "cpu") {
      idx = faiss::read_index("index.faiss");
    } else {
      auto options = new faiss::gpu::GpuMultipleClonerOptions();
      options->shard = true;
      idx = faiss::gpu::index_cpu_to_gpu_multiple(
          res, devs, faiss::read_index("index.faiss"), options);
    }

    // Containers to hold the search results
    std::vector<faiss::idx_t> nns(top_k * n_query);
    std::vector<float> dis(top_k * n_query);

    // Perform the search
    auto s = std::chrono::high_resolution_clock::now();
    idx->search(n_query, data_query, top_k, dis.data(), nns.data());
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Search: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    // Delete the search dataset
    delete[] data_query;
  }

  return 0;
}
