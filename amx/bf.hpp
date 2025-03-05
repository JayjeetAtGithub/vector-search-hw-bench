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
  int32_t _nq;
  int32_t _nl;

  dnnl::engine engine;
  dnnl::stream stream;
  std::vector<bf16> _distances;

public:
  void init_onednn() {
    engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream = dnnl::stream(engine);
  }

  BruteForceSearch(int32_t dim, int32_t nq, int32_t nl) : _dim(dim), _nq(nq), _nl(nl) {
    init_onednn();
    if (!is_amxbf16_supported()) {
      std::cout << "Intel AMX unavailable" << std::endl;
    }
    _distances.resize((int64_t)_nq * (int64_t)_nl);
  }

  std::vector<std::vector<int>> search_ip_amx(
    std::vector<bf16> &queries, std::vector<bf16> &dataset, int32_t top_k) {
    
    amx_inner_product(
      _nq, _nl, _dim, queries, dataset, _distances, engine, stream
    );
    
    std::unordered_map<
      int32_t, 
      std::priority_queue<
        std::pair<int32_t, float>, 
        std::vector<std::pair<int32_t, float>>, 
        Comp
      >>
    m;
    
    // #pragma omp parallel for
    // for (int32_t i = 0; i < _nq; i++) {
    //     std::priority_queue<std::pair<int32_t, float>> local_queue;
    //     for (int32_t j = 0; j < _nl; j++) {
    //         int64_t offset = (int64_t)i * (int64_t)_nl + (int64_t)j;
    //         float dist = _distances[offset];

    //         if (local_queue.size() < top_k) {
    //             local_queue.push({j, dist});
    //         } else {
    //             if (local_queue.top().second > dist) {
    //                 local_queue.pop();
    //                 local_queue.push({j, dist});
    //             }
    //         }
    //     }
    //     #pragma omp critical
    //     {
    //         while (!local_queue.empty()) {
    //             m[i].push(local_queue.top());
    //             local_queue.pop();
    //         }
    //     }
    // }

    // serial version

    for (int32_t i = 0; i < _nq; i++) {
      for (int32_t j = 0; j < _nl; j++) {
        int64_t offset = (int64_t)i * (int64_t)_nl + (int64_t)j;
        float dist = _distances[offset];

        if (m[i].size() < top_k) {
          m[i].push({j, dist});
        } else {
          if (m[i].top().second > dist) {
            m[i].pop();
            m[i].push({j, dist});
          }
        }
      }
    }

    std::vector<std::vector<int>> results(
      _nq, std::vector<int>(top_k)
    );

    for (int32_t i = 0; i < _nq; i++) {
      int32_t k_idx = top_k - 1;
      while (k_idx >= 0) {
        results[i][k_idx--] = m[i].top().first;
        m[i].pop();
      }
    }

    return results;
  }
};
