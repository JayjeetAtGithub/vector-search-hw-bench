#pragma once

#include <chrono>
#include <immintrin.h>
#include <unordered_map>
#include <stdfloat>

#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;
using bf16 = std::bfloat16_t;

static bool is_amxbf16_supported() {
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid"
                       : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                       : "a"(7), "c"(0));
  return edx & (1 << 22);
}

static void amx_inner_product(int32_t const &n, int32_t const &oc,
                              int32_t const &ic, std::vector<bf16> &s, std::vector<bf16> &w,
                              std::vector<bf16> &res, dnnl::engine &engine, 
                              dnnl::stream &stream) {
  dnnl::memory::dims s_dims = {n, ic};
  dnnl::memory::dims w_dims = {oc, ic};
  dnnl::memory::dims dst_dims = {n, oc};

  auto s_md = dnnl::memory::desc(s_dims, dt::bf16, tag::ab);
  auto w_md = dnnl::memory::desc(w_dims, dt::bf16, tag::ab);
  auto dst_md = dnnl::memory::desc(dst_dims, dt::bf16, tag::ab);
  
  auto s_mem = dnnl::memory(s_md, engine);
  auto w_mem = dnnl::memory(w_md, engine);

  write_to_dnnl_memory(s.data(), s_mem);
  write_to_dnnl_memory(w.data(), w_mem);
  
  auto pd = dnnl::inner_product_forward::primitive_desc(
      engine, dnnl::prop_kind::forward_training, s_md, w_md, dst_md);
  auto dst_mem = dnnl::memory(pd.dst_desc(), engine);

  auto prim = dnnl::inner_product_forward(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, s_mem});
  args.insert({DNNL_ARG_WEIGHTS, w_mem});
  args.insert({DNNL_ARG_DST, dst_mem});
  prim.execute(stream, args);
  stream.wait();

  auto st = std::chrono::high_resolution_clock::now();
  read_from_dnnl_memory(res.data(), dst_mem);
  auto en = std::chrono::high_resolution_clock::now();
  std::cout << "[TIME] AMX: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count()
            << " ms" << std::endl;
}
