# Vector Search on CPUs, GPUs, and On-Chip Accelerators

We study the performance of vector search algorithms such as IVF, HNSW, and Flat on 
different hardware architectures such as Sapphire Rapids CPU, A100 GPUs, and Intel AMX
accelerators. 

## Cloning the Repository

```bash
git clone --recursive https://github.com/JayjeetAtGithub/vector-search-gpu
```

## Downloading Dataset

We use the Yandex Text-to-Image dataset from [here](https://big-ann-benchmarks.com/neurips21.html).
It has `float32` elements in vectors of 200 dimensions and built using IP as the distance metric.

```bash
mkdir -p /workspace/dataset/t2i
cd /workspace/dataset/t2i
wget -O dataset.bin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
wget -O query.bin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
```

## Downloading Indexes

```bash
cd src/
aws s3 cp --recursive s3://cpu-faiss-indexes .
```

## Installing FAISS on CPU & GPU

```bash
# for Sapphire Rapids CPU
./install_faiss_cpu_spr.sh

# for other CPU
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
