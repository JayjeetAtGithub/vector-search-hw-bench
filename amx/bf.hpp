#include <queue>
#include <vector>

#include "distance.hpp"

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

  std::pair<int32_t, int32_t> shape() {
    return std::make_pair(_dataset.size(), _dataset[0].size());
  }

  std::vector<std::vector<float>> search_ip_amx(std::vector<std::vector<float>> queries, int32_t top_k) {
    std::vector<std::vector<float>> results(queries.size(),
                                            std::vector<float>(top_k, 0.0f));
    std::unordered_map<int32_t, std::priority_queue<float, std::vector<float>,
                                                    std::greater<float>>>
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
        queries.size(), _dataset.size(), _dim, mat_a.data(), mat_b.data(), engine, stream, debug);

    for (int32_t i = 0; i < distances.size(); i++) {
        for (int32_t j = 0; j < distances[0].size(); j++) {
            map[i].push(distances[i][j]);
        }
    }

    for (int i = 0; i < queries.size(); i++) {
      int32_t k_idx = 0;
      while (k_idx < top_k) {
        results[i][k_idx++] = map[i].top();
        map[i].pop();
      }
    }
    return results;
  }
};
