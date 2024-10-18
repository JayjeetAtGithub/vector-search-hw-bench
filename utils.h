#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

float *file_read(const char *fname, int64_t *n, int64_t *d, int64_t limit) {
  FILE *f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", fname);
    perror("");
    abort();
  }

  int64_t e;

  int64_t N;
  e = fread(&N, sizeof(uint32_t), 1, f);
  *n = std::min(N, limit);

  int64_t D;
  e = fread(&D, sizeof(uint32_t), 1, f);
  *d = D;

  int64_t len = std::min(N, limit) * D;
  float *v = new float[len];
  e = fread(v, sizeof(float), len, f);
  std::cout << "Read " << e << " elements" << std::endl;

  fclose(f);
  return v;
}

void preview_dataset(float *xb) {
  for (int64_t i = 0; i < 5; i++) {
    for (int64_t j = 0; j < 10; j++) {
      std::cout << xb[i * 10 + j] << " ";
    }
    std::cout << std::endl;
  }
}

void read_dataset(const char *filename, float *&xb, int64_t *d, int64_t *n,
                  int64_t limit) {
  xb = file_read(filename, d, n, limit);
}

void write_vector(const char *filename, int64_t *data, int64_t size) {
  FILE *f = fopen(filename, "w");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", filename);
    perror("");
    abort();
  }
  int64_t e = fwrite(data, sizeof(int64_t), size, f);
  fclose(f);
}

std::vector<int64_t> read_vector(const char *filename, int64_t size) {
  FILE *f = fopen(filename, "r");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", filename);
    perror("");
    abort();
  }
  std::vector<int64_t> data(size);
  int64_t e = fread(data.data(), sizeof(int64_t), size, f);
  fclose(f);
  return data;
}
