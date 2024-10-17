# Vector Search on GPU

**NOTE**: In HNSW and IVF implementations, the default construction parameters are used. Only the search parameters can be changed. The default construction parameters are as follows:

```bash
# HNSW
M = 32
efConstruction = 40

# IVF-Flat
n_list = int64_t(4 * std::sqrt(n_learn))
```

# System Requirements

* cmake >= 3.29
* g++ >= 13.1.0
* CUDA >= 12.4

# Building from source

```bash
mkdir build/
cd build/
cmake ..
make
```
