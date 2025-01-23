#include <fstream>
#include <iostream>
#include <thread>

void preview_dataset(std::vector<float> xb) {
  for (int64_t i = 0; i < 5; i++) {
    for (int64_t j = 0; j < 10; j++) {
      std::cout << xb[i * 10 + j] << " ";
    }
    std::cout << std::endl;
  }
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

std::vector<float> read_bin_dataset(std::string fname, int64_t *d, int64_t limit) {
  // Read datafile in
  std::ifstream datafile(fname, std::ifstream::binary);
  uint32_t dim_uint32;
  datafile.read((char *)&dim_uint32, sizeof(uint32_t));
  int64_t dim = (int64_t)dim_uint32;
  *d = dim;
  printf("Read in file - N:%li, dim:%li\n", N, dim);
  std::vector<float> data;
  data.resize((size_t)limit * (size_t)dim);
  datafile.read(reinterpret_cast<char *>(data.data()),
                (size_t)limit * (size_t)dim * sizeof(float));
  datafile.close();

  return data;
}
