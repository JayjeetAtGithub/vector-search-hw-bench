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
T *file_read(const char *fname, int64_t *n, int64_t *d, int64_t limit) {
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
  T *v = new T[len];
  e = fread(v, sizeof(T), len, f);
  std::cout << "Read " << e << " elements" << std::endl;

  fclose(f);
  return v;
}

template <typename T> void preview_dataset(T *xb) {
  for (int64_t i = 0; i < 5; i++) {
    for (int64_t j = 0; j < 10; j++) {
      std::cout << xb[i * 10 + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
void read_dataset(const char *filename, T *&xb, int64_t *d, int64_t *n,
                  int64_t limit) {
  xb = file_read<T>(filename, d, n, limit);
}
