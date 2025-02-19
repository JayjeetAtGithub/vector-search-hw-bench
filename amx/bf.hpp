#pragma once

#include <queue>
#include <vector>
#include <iostream>

#include "distance.hpp"

struct Comp {
  static bool operator()(const std::pair<int, float> &a, const std::pair<int, float> &b) {
    return a.second < b.second;
  }
};

class BruteForceSearch {
  int32_t _dim;
  std::vector<float> _dataset;

  dnnl::engine engine;
  dnnl::stream stream;

public:
  void init_onednn() {
    engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream = dnnl::stream(engine);
  }

  BruteForceSearch(int32_t dim) : _dim(dim) {
    init_onednn();
    if (!is_amxbf16_supported()) {
      std::cout << "Intel AMX unavailable" << std::endl;
    }
  }

  void add(std::vector<float> dataset) { _dataset = dataset; }

  void search_ip_amx(std::vector<float> queries, int32_t top_k) {
    std::vector<bf16> results(queries.size() * top_k);
    std::unordered_map<
      int32_t, 
      std::priority_queue<
        std::pair<int, float>, 
        std::vector<std::pair<int, float>>, 
        Comp
      >
    map;

    std::vector<bf16> mat_a(queries.size());
    std::vector<bf16> mat_b(_dataset.size());

    for (int32_t i = 0; i < queries.size(); i++) {
      mat_a[i] = bf16(queries[i]);
    }
    
    for (int32_t i = 0; i < _dataset.size(); i++) {
      mat_b[i] = bf16(_dataset[i]);
    }

    amx_inner_product(
      queries.size(), _dataset.size(), _dim, mat_a.data(), 
      mat_b.data(), results.data(), engine, stream
    );

    for (int32_t i = 0; i < results.size(); i++) {
      for (int32_t j = 0; j < results[0].size(); j++) {
        map[i].push({j, distances[i][j]});
      }
    }

    // for (int i = 0; i < queries.size(); i++) {
    //   int32_t k_idx = 0;
    //   while (k_idx < top_k) {
    //     results[i][k_idx++] = map[i].top();
    //     map[i].pop();
    //   }
    // }
    // return results;
  }
};
