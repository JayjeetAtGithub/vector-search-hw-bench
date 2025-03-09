# Vector Search on CPUs, GPUs, and On-Chip Accelerators

We study the performance of vector search algorithms such as IVF, HNSW, and Flat on 
different hardware architectures such as Sapphire Rapids CPU, A100 GPUs, and Intel AMX
accelerators. 

# Tested Hardware

For CPU/AMX experiments, we have tested on UC Santa Cruz's Sapphire Rapids machine.
For GPU experiments, we have tested the code on Lambda clouds `gpu_1x_a100_sxm4` machine.

# Dataset

We use the Yandex Text-to-Image dataset from [here](https://big-ann-benchmarks.com/neurips21.html).
It has float32 elements in vectors of 200 dimensions and built using IP as the distance metric.

Learn dataset: https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin

Query dataset: https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin

Groundtruth: https://storage.yandexcloud.net/yandex-research/ann-datasets/t2i_new_groundtruth.public.100K.bin

# Downloading Indexes

```bash
cd src/
aws s3 cp --recursive s3://cpu-faiss-indexes .
```

## Installing FAISS on CPU & GPU

```bash
# for Sapphire Rapids CPU
./install_faiss_cpu.sh

# for NVIDIA GPU
./install_faiss_gpu.sh
```

## Machine Setup

### Updating `cmake`

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.sh
chmod +x cmake-3.30.5-linux-x86_64.sh
./cmake-3.30.5-linux-x86_64.sh
cd cmake-3.30.5-linux-x86_64
sudo cp -r bin/* /usr/local/bin/
sudo cp -r doc/* /usr/local/doc/
sudo cp -r man/* /usr/local/man/
sudo cp -r share/* /usr/local/share/
cmake --version
```
