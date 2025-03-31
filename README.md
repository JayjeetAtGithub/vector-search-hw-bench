# Vector Search on CPUs, GPUs, and On-Chip Accelerators

We study the performance of vector search algorithms such as IVF, HNSW, and Flat on 
different hardware architectures such as Sapphire Rapids CPU, A100 GPUs, and Intel AMX
accelerators. 

## Preferred Machines for Benchmarking

1. CPU/AMX: r7iz.metal-32xl
2. GPU: Lambda H100 80GB SXM5

## Cloning the Repository

```bash
git clone --recursive git@github.com:JayjeetAtGithub/vector-search-hw-bench
```

## Downloading Dataset

We use the Yandex Text-to-Image dataset from [here](https://big-ann-benchmarks.com/neurips21.html).
It has `float32` elements in vectors of 200 dimensions and built using IP as the distance metric.

```bash
mkdir -p /workspace/dataset/t2i
cd /workspace/dataset/t2i
axel -o dataset.bin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
axel -o query.bin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
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

### Installing Docker

```bash
curl -o- https://get.docker.com | bash
```

### Downloading oneAPI image

```bash
docker pull intel/oneapi:2025.1.0-0-devel-ubuntu24.04
```

### Updating `cmake`

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.sh
chmod +x cmake-3.30.5-linux-x86_64.sh
./cmake-3.30.5-linux-x86_64.sh
cd cmake-3.30.5-linux-x86_64
sudo cp -r bin/*   /usr/local/bin/
sudo cp -r doc/*   /usr/local/doc/
sudo cp -r man/*   /usr/local/man/
sudo cp -r share/* /usr/local/share/
sudo cp -r bin/*   /usr/bin/
sudo cp -r doc/*   /usr/doc/
sudo cp -r man/*   /usr/man/
sudo cp -r share/* /usr/share/
cmake --version
```
