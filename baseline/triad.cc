#include <iostream>
#include <omp.h>
#include <cstdlib>

int main() {
    const size_t N = 1'000'000'000;
    const int iterations = 200;
    const float scalar = 3.0f;

    float* A = (float*) malloc(N * sizeof(float));
    float* B = (float*) malloc(N * sizeof(float));
    float* C = (float*) malloc(N * sizeof(float));

    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed\n";
        return 1;
    }

    // Initialize B and C
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        B[i] = 1.0f;
        C[i] = 2.0f;
    }

    // Warm-up run (helps cache and avoid cold start)
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        A[i] = B[i] + scalar * C[i];
    }

    std::cout << "Starting sustained STREAM Triad...\n";

    // Measure sustained bandwidth
    double start = omp_get_wtime();

    for (int iter = 0; iter < iterations; iter++) {
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            A[i] = B[i] + scalar * C[i];
        }
    }

    double end = omp_get_wtime();
    double total_time = end - start;

    double total_bytes_read = 3.0 * N * sizeof(float) * iterations;
    double sustained_bandwidth_read = total_bytes_read / total_time / 1e9; // GB/s

    double total_bytes_write = 1.0 * N * sizeof(float) * iterations;
    double sustained_bandwidth_write = total_bytes_write / total_time / 1e9; // GB/s

    std::cout << "Total time for " << iterations << " iterations: " << total_time << " seconds\n";
    std::cout << "Read bandwidth: " << sustained_bandwidth_read << " GB/s\n";
    std::cout << "Write bandwidth: " << sustained_bandwidth_write << " GB/s\n";
    std::cout << "Total bandwidth: " << (sustained_bandwidth_read + sustained_bandwidth_write) << " GB/s\n";

    free(A); free(B); free(C);
    return 0;
}
