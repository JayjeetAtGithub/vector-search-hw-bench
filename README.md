# Vector Search on GPU

## Installing FAISS
```bash
mkdir build/
cd build/
cmake -DBLA_VENDOR=OpenBLAS -DMKL_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.a -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_RAFT=ON -DBUILD_TESTING=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_BUILD_TYPE=Release ..
sudo make -j$(nproc) install
```
