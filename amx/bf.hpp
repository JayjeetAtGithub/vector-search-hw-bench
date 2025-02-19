#pragma once

#include <queue>
#include <vector>
#include <iostream>
#include <map>

#include "distance.hpp"

struct Comp {
  static bool operator()(const std::pair<int, float> &a, const std::pair<int, float> &b) {
    return a.second < b.second;
  }
};

class BruteForceSearch {
  int32_t _dim;

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

  void search_ip_amx(
    std::vector<float> &queries, int32_t nq,
    std::vector<float> &dataset, int32_t nl, int32_t top_k) {

    std::vector<bf16> distances(nq * top_k);
    std::unordered_map<
      int32_t, 
      std::priority_queue<
        std::pair<int, float>, 
        std::vector<std::pair<int, float>>, 
        Comp
      >>
    m;

    std::vector<bf16> mat_a(queries.size());
    std::vector<bf16> mat_b(dataset.size());

    for (int32_t i = 0; i < queries.size(); i++) {
      mat_a[i] = bf16(queries[i]);
    }
    
    for (int32_t i = 0; i < dataset.size(); i++) {
      mat_b[i] = bf16(dataset[i]);
    }

    amx_inner_product(
      nq, nl, _dim, mat_a, mat_b, distances, engine, stream
    );

    for (int32_t i = 0; i < nq; i++) {
      for (int32_t j = 0; j < _dim; j++) {
        m[i].push({j, distances[i * _dim + j]});
      }
    }

    std::vector<std::vector<int>> results(
      nq, std::vector<int>(top_k)
    );

    for (int i = 0; i < nq; i++) {
      int32_t k_idx = 0;
      while (k_idx < top_k) {
        results[i][k_idx++] = m[i].top().first;
        m[i].pop();
      }
    }
    // return results;
  }
};
