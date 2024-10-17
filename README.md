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

* Ubuntu 22.04+
* Cmake >= 3.29
* G++ >= 13.1.0
* CUDA >= 12.4

# Building from source

```bash
mkdir build/
cd build/
cmake ..
make
```

## Machine Setup

### Updating `gcc` and `g++`

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100
sudo update-alternatives --config gcc
sudo update-alternatives --config g++
gcc --version
g++ --version
```

### NVIDIA Drivers and CUDA Toolkit
