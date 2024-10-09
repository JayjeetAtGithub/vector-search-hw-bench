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

template <typename T>
T *file_read(const char *fname, uint32_t *n, uint32_t *d, uint32_t limit) {
  FILE *f = fopen(fname, "r");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", fname);
    perror("");
    abort();
  }

  int e;

  uint32_t N;
  e = fread(&N, sizeof(uint32_t), 1, f);
  *n = std::min(N, limit);

  uint32_t D;
  e = fread(&D, sizeof(uint32_t), 1, f);
  *d = D;

  uint32_t len = std::min(N, limit) * D;
  T *v = new T[len];
  e = fread(v, sizeof(T), len, f);

  fclose(f);
  return v;
}

void preview_dataset(float *xb) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 10; j++) {
      std::cout << xb[i * 10 + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void read_dataset2(const char *filename, T *&xb, uint32_t *d, uint32_t *n,
                   uint32_t limit) {
  xb = file_read<T>(filename, d, n, limit);
}
