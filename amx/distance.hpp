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

// static void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
//   dnnl::engine eng = mem.get_engine();
//   int32_t size = mem.get_desc().get_size();
//   if (!handle)
//     throw std::runtime_error("handle is nullptr.");
//   uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
//   if (!src)
//     throw std::runtime_error("get_data_handle returned nullptr.");
//   for (int32_t i = 0; i < size; ++i) {
//     ((uint8_t *)handle)[i] = src[i];
//   }
// }

// static void write_to_dnnl_memory(void const *handle, dnnl::memory &mem) {
//   dnnl::engine eng = mem.get_engine();
//   int32_t size = mem.get_desc().get_size();
//   if (!handle)
//     throw std::runtime_error("handle is nullptr.");
//   uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
//   if (!dst)
//     throw std::runtime_error("get_data_handle returned nullptr.");
//   for (int32_t i = 0; i < size; ++i) {
//     dst[i] = ((uint8_t *)handle)[i];
//   }
// }

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

  std::cout << "s.size(): " << s.size() << std::endl;

  write_to_dnnl_memory(s.data(), s_mem);
  write_to_dnnl_memory(w.data(), w_mem);
  
  std::cout << "s_mem.get_desc().get_size(): " << s_mem.get_desc().get_size() << std::endl;
  
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

  read_from_dnnl_memory(res.data(), dst_mem);
}
