#pragma once

#include <chrono>
#include <immintrin.h>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

using tag = dnnl::memory::format_tag;
using dt = dnnl::memory::data_type;

static bool is_amxbf16_supported() {
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid"
                       : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                       : "a"(7), "c"(0));
  return edx & (1 << 22);
}

static void amx_inner_product(int32_t const &n, int32_t const &oc,
                              int32_t const &ic, std::vector<float> &s, std::vector<float> &w,
                              std::vector<float> &res, dnnl::engine &engine, 
                              dnnl::stream &stream) {
  dnnl::memory::dims s_dims = {n, ic};
  dnnl::memory::dims w_dims = {oc, ic};
  dnnl::memory::dims dst_dims = {n, oc};

  auto s_in_md = dnnl::memory::desc(s_dims, dt::f32, tag::ab);
  auto w_in_md = dnnl::memory::desc(w_dims, dt::f32, tag::ab);
  auto dst_out_md = dnnl::memory::desc(dst_dims, dt::f32, tag::ab);
  auto s_in_mem = dnnl::memory(s_in_md, engine);
  auto w_in_mem = dnnl::memory(w_in_md, engine);
  write_to_dnnl_memory(s.data(), s_in_mem);
  write_to_dnnl_memory(w.data(), w_in_mem);

  auto s_md = dnnl::memory::desc(s_dims, dt::bf16, tag::any);
  auto w_md = dnnl::memory::desc(w_dims, dt::bf16, tag::any);

  auto pd = dnnl::inner_product_forward::primitive_desc(
      engine, dnnl::prop_kind::forward_training, s_md, w_md, dst_out_md);
  
  auto s_mem = dnnl::memory(pd.src_desc(), engine);
  dnnl::reorder(s_in_mem, s_mem).execute(stream, s_in_mem, s_mem);
  auto w_mem = dnnl::memory(pd.weights_desc(), engine);
  dnnl::reorder(w_in_mem, w_mem).execute(stream, w_in_mem, w_mem);
  auto dst_mem = dnnl::memory(pd.dst_desc(), engine);
  
  auto prim = dnnl::inner_product_forward(pd);
  std::unordered_map<int32_t, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, s_mem});
  args.insert({DNNL_ARG_WEIGHTS, w_mem});
  args.insert({DNNL_ARG_DST, dst_mem});
  prim.execute(stream, args);
  stream.wait();

  read_from_dnnl_memory(res.data(), dst_mem);
}
