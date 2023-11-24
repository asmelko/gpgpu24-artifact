### Parameters
- `algorithmType` = `topk`, `projection`, `esom`
- `algo` = different for different types
- `N` = 1M
- `dim` = 4, 8, 16, 32, 64
- `gridSize` = 128, 256, 512, 1024
- `k` = 8, 16, 32, 64 (pro topk, pro projection jsou to hodnoty -1, aby to sedelo)

### Algorithms

**topk**

- `serial`
- `cuda_base`
	- `cudaBlockSize` = 8, 16, 32, 64, 128, 256
	- optimal size should be around 64, but for higher dimensions (>10), the optimum shifts towards 16 (probably better caching)
- `cuda_shm`
	- `cudaBlockSize` = 64, 128, 256
	- `cudaSharedMemorySize` = 4k, 8k, 16k, 32k
- `cuda_radixsort`
	- `cudaBlockSize` = 64, 128, 256, 512, 1024
- `cuda_2dinsert`
	- `itemsPerBlock` = 8, 16, 32
	- `itemsPerThread` = 1, 2, 4, 8, 16 (< itemsPerBlock)
- `cuda_bitonicopt`
	- `cudaBlockSize` = 64, 128, 256, 512, 1024

**projection**
- `serial`
- `cuda_base`
	- `cudaBlockSize` = 64, 128, 256, 512, 1024
- `cuda_block_multi`
- `cuda_block_aligned`
- `cuda_block_aligned_register`

all CUDA algorithms are executed with
- `cudaBlockSize` = 8, 16, 32, 64, 128, 256
- `groupsPerBlock` = 1, 2, 3, 4


### Results

Path: `hostname`/`algorithmType`/`algo`-`N`-`dim`-`gridSize`-`k`.csv
CSV columns
- `algorithmType`
- `algo`
- `N`
- `dim`
- `gridSize`
- `k`
- `cudaBlockSize`
- `cudaSharedMemorySize`
- `groupsPerBlock`
- `itemsPerBlock`
- `itemsPerThread`
- `initTime` [ms]
- `kernelTime` [ms] -- this is the important number !!!

