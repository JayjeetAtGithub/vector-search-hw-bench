cmake_minimum_required(VERSION 3.28)

project(vector-search)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -Og -g3 -fopenmp -fno-omit-frame-pointer")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(run_faiss run_faiss.cc)
target_link_libraries(run_faiss faiss)
