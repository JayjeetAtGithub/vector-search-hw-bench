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

  std::vector<std::vector<int>> search_ip_amx(
    std::vector<bf16> &queries, int32_t nq,
    std::vector<bf16> &dataset, int32_t nl, int32_t top_k) {

    std::vector<bf16> distances;
    distances.resize((int64_t)nq * (int64_t)nl);

    std::unordered_map<
      int32_t, 
      std::priority_queue<
        std::pair<int32_t, float>, 
        std::vector<std::pair<int32_t, float>>, 
        Comp
      >>
    m;

    auto s = std::chrono::high_resolution_clock::now();
    amx_inner_product(
      nq, nl, _dim, queries, dataset, distances, engine, stream
    );
    auto e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] AMX Inner Product: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    s = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int32_t i = 0; i < nq; i++) {
        std::priority_queue<std::pair<int32_t, float>> local_queue;
        for (int32_t j = 0; j < nl; j++) {
            int64_t offset = (int64_t)i * (int64_t)nl + (int64_t)j;
            float dist = distances[offset];

            if (local_queue.size() < top_k) {
                local_queue.push({j, dist});
            } else {
                if (local_queue.top().second > dist) {
                    local_queue.pop();
                    local_queue.push({j, dist});
                }
            }
        }
        #pragma omp critical
        {
            while (!local_queue.empty()) {
                m[i].push(local_queue.top());
                local_queue.pop();
            }
        }
    }
    e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Finding TopK: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    s = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> results(
      nq, std::vector<int>(top_k)
    );

    for (int32_t i = 0; i < nq; i++) {
      int32_t k_idx = top_k - 1;
      while (k_idx >= 0) {
        results[i][k_idx--] = m[i].top().first;
        m[i].pop();
      }
    }
    e = std::chrono::high_resolution_clock::now();
    std::cout
        << "[TIME] Assembling Results: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
        << " ms" << std::endl;

    return results;
  }
};
