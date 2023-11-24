#ifdef __INTELLISENSE__
#define __CUDACC__
#endif 

#include "topk.cuh"

#include <climits>
#include <cstdint>

#include "cooperative_groups.h"
#include "cub/block/block_radix_sort.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitonic.cuh"

namespace cg = cooperative_groups;

template <typename F = float>
constexpr F valueMax;

template <> constexpr float valueMax<float> = FLT_MAX;
template <> constexpr double valueMax<double> = DBL_MAX;

template <typename F = float>
__inline__ __device__ F distance(const F* __restrict__ lhs, const F* __restrict__ rhs, const std::uint32_t dim)
{
	F sum = (F)0.0;
	for (std::uint32_t d = 0; d < dim; ++d) {
		F diff = *lhs++ - *rhs++;
		sum += diff * diff;
	}
	return sum; // squared euclidean
}

template <typename F = float>
__inline__ __device__ void bubbleUp(typename TopkProblemInstance<F>::Result* const __restrict__ topK, std::uint32_t idx)
{
	while (idx > 0 && topK[idx - 1].distance > topK[idx].distance) {
		const typename TopkProblemInstance<F>::Result tmp = topK[idx];
		topK[idx] = topK[idx - 1];
		topK[idx - 1] = tmp;
		--idx;
	}
}

/**
 * Each thread iterates over whole point grid and computes kNN for a specified point
 * using insertion sort in global memory.
 */
template <typename F>
__global__ void topkBaseKernel(const F* __restrict__ points, const F* const __restrict__ grid, typename TopkProblemInstance<F>::Result* __restrict__ topKs,
							   const std::uint32_t dim, const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k)
{
	// assign correct point and topK pointers for a thread
	{
		const std::uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		if (pointIdx >= n)
			return;

		topKs = topKs + pointIdx * k;
		points = points + pointIdx * dim;
	}

	// iterate over grid points
	{
		std::uint32_t gridIdx = 0;

		for (; gridIdx < k; ++gridIdx) 
		{
			topKs[gridIdx] = { distance<F>(points, grid + gridIdx * dim, dim), gridIdx };
			bubbleUp<F>(topKs, gridIdx);
		}

		for (; gridIdx < gridSize; ++gridIdx) 
		{
			F dist = distance<F>(points, grid + gridIdx * dim, dim);

			if (topKs[k - 1].distance > dist) {
				topKs[k - 1] = { dist, gridIdx };
				bubbleUp<F>(topKs, k - 1);
			}
		}
	}
}

// runner wrapped in a class
template <typename F>
void TopkBaseKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = (in.n + exec.blockSize - 1) / exec.blockSize;
	topkBaseKernel<F><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.gridSize, in.k);
}


/**
 * Each thread iterates over batches of grid points loaded in shared memory
 * and computes kNN for a specified point using insertion sort in global memory.
 * We expect that k < sharedSize.
 */
template <typename F>
__global__ void topkThreadSharedKernel(const F* __restrict__ points, const F* const __restrict__ grid,
									   typename TopkProblemInstance<F>::Result* __restrict__ topKs, const std::uint32_t dim, const std::uint32_t n,
									   const std::uint32_t gridSize, const std::uint32_t k, const std::uint32_t sharedSize)
{
	extern __shared__ char sharedMemory[];
	F* const gridCache = reinterpret_cast<F*>(sharedMemory);

	const std::uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// assign correct point and topK pointers for a thread
	{
		topKs = topKs + pointIdx * k;
		points = points + pointIdx * dim;
	}

	// iterate over grid points
	{
		std::uint32_t gridOffset = 0;
		while (gridOffset < gridSize) 
		{
			const std::uint32_t batchLimit = min(gridOffset + sharedSize, gridSize);

			// store to shared
			{
				const std::uint32_t batchSize = (batchLimit - gridOffset) * dim;
				const F* const __restrict__ batchPtr = grid + gridOffset * dim;
				for (std::uint32_t i = threadIdx.x; i < batchSize; i += blockDim.x) {
					gridCache[i] = batchPtr[i];
				}
				// TODO -- tohle neni dobry ani z pohledu coalesced loadu ani z pohledu zapisu do bank.
				//for (std::uint32_t i = gridOffset + threadIdx.x; i < batchLimit; i += blockDim.x)
				//	memcpy(gridCache + (i - gridOffset) * dim, grid + i * dim, sizeof(F) * dim);

				__syncthreads();
			}

			// iterate over batch of grid points loaded into shared memory
			if (pointIdx < n) 
			{
				std::uint32_t gridIdx = gridOffset;

				for (; gridIdx < k; ++gridIdx) {
					topKs[gridIdx] = { distance<F>(points, gridCache + gridIdx * dim, dim), gridIdx };
					bubbleUp<F>(topKs, gridIdx);
				}

				for (; gridIdx < batchLimit; ++gridIdx) 
				{
					F dist = distance<F>(points, gridCache + (gridIdx - gridOffset) * dim, dim);

					if (topKs[k - 1].distance > dist) {
						topKs[k - 1] = { dist, gridIdx };
						bubbleUp<F>(topKs, k - 1);
					}
				}
			}

			gridOffset += sharedSize;
			__syncthreads();
		}
	}
}

// runner wrapped in a class
template <typename F>
void TopkThreadSharedKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = (in.n + exec.blockSize - 1) / exec.blockSize;
	std::uint32_t sharedSize = exec.sharedMemorySize / (in.dim * sizeof(F));

	topkThreadSharedKernel<F>
		<<<blockCount, exec.blockSize, exec.sharedMemorySize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.gridSize, in.k, sharedSize);
}


/**
 * Whole grid block iterate over grid points to compute kNN of a specified point.
 * In one block step over grid points, a block thread aggregates radixThreadWork results.
 * These are then collectively sorted using block radix sort from CUB library.
 * First k of sorted results are stored to temporary kNN array in shared memory.
 * Then serial insertion sort is performed to account to the whole-loop-wide kNN array in shared memory.  
 * We expect that k < blockDim.x * radixThreadWork.
 */
template <typename F, int blockDimX, std::uint32_t radixThreadWork>
__global__ void topkBlockRadixSortKernel(const F* const __restrict__ points, const F* const __restrict__ grid,
										 typename TopkProblemInstance<F>::Result* const __restrict__ topKs, const std::uint32_t dim, const std::uint32_t n,
										 const std::uint32_t gridSize, const std::uint32_t k)
{
	extern __shared__ char sharedMemory[];

	// static shared memory needed for BlockRadixSort
	typedef cub::BlockRadixSort<F, blockDimX, radixThreadWork, std::uint32_t> BlockRadixSort;
	typename BlockRadixSort::TempStorage* const tempStorage = reinterpret_cast<typename BlockRadixSort::TempStorage*>(sharedMemory);

	F* const pointCache = reinterpret_cast<F*>(tempStorage + 1);
	typename TopkProblemInstance<F>::Result* const topKCache = reinterpret_cast<typename TopkProblemInstance<F>::Result*>(pointCache + dim);
	typename TopkProblemInstance<F>::Result* const topKCacheMerging = topKCache + k;


	// thread local caches
	F threadDists[radixThreadWork];
	std::uint32_t threadIdxs[radixThreadWork];

	// load the point into shared memory
	// clear shared topK cache
	{
		for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) 
			pointCache[i] = points[blockIdx.x * dim + i];
		for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) 
			topKCache[i].distance = valueMax<F>;

		__syncthreads();
	}

	// iterate over the grid points
	std::uint32_t gridOffset = 0;
	while (gridOffset < gridSize) 
	{
		// fill local cache of a thread
		{
			const std::uint32_t gridIdx = gridOffset + radixThreadWork * threadIdx.x;
			const std::uint32_t gridLimit = min(radixThreadWork, gridSize < gridIdx ? 0 : gridSize - gridIdx);
			std::uint32_t localCacheIdx = 0;

			for (; localCacheIdx < gridLimit; ++localCacheIdx) {
				threadDists[localCacheIdx] = distance<F>(pointCache, grid + (gridIdx + localCacheIdx) * dim, dim);
				threadIdxs[localCacheIdx] = gridIdx + localCacheIdx;
			}
			for (; localCacheIdx < radixThreadWork; ++localCacheIdx)
				threadDists[localCacheIdx] = valueMax<F>;
		}

		// collectively sort the keys (dists) and values (idxs) among block threads
		{
			__syncthreads(); // is here to prevent overwriting topKCacheMerging
							 // logically, this call can be places after insertion sort, but
							 // in this case we are letting other threads (warps) work on distance computation 
							 // while thread0 performs insertion sort
							 // may be redundant if BlockRadixSort uses __syncthreads
			BlockRadixSort(*tempStorage).Sort(threadDists, threadIdxs);
		}

		// store sorted values to shared topKs array from first k/radixThreadWork threads
		{
			const std::uint32_t topKIdx = threadIdx.x * radixThreadWork;
			const std::uint32_t topKLimit = min(radixThreadWork, k < topKIdx ? 0 : k - topKIdx);

			for (std::uint32_t i = 0; i < topKLimit; i++)
				topKCacheMerging[topKIdx + i] = { threadDists[i], threadIdxs[i] };

			__syncthreads();
		}

		// merge topKCacheMerging into topKCache using serial insertion sort
		if (threadIdx.x == 0) 
		{
			for (std::uint32_t i = 0; i < k; i++) {
				if (topKCache[k - 1].distance > topKCacheMerging[i].distance) {
					topKCache[k - 1] = topKCacheMerging[i];
					bubbleUp<F>(topKCache, k - 1);
				}
				else
					break; // when the current top of topKCacheMerging is higher than the bottom of 
						   // topKCache, all other elements of topKCacheMerging are higher as well
			}
		}

		gridOffset += blockDim.x * radixThreadWork;
	}

	// finally, write topKCache to global memory
	{
		__syncthreads();

		for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) 
			topKs[blockIdx.x * k + i] = topKCache[i];
	}
}

// runner wrapped in a class
template <typename F>
void TopkBlockRadixSortKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	unsigned sharedMemory = in.dim * sizeof(F) +										// for point
							in.k * sizeof(typename TopkProblemInstance<F>::Result) * 2; // for topK cache and topK merging cache

#define CASE_TopkBlockRadixSortKernelInner(THREADS, WORK)                                                                                            \
	case WORK:                                                                                                                                       \
		sharedMemory += sizeof(typename cub::BlockRadixSort<F, THREADS, WORK, std::uint32_t>::TempStorage); /* for radix sort */                     \
		if (sharedMemory > exec.sharedMemorySize)                                                                                                    \
			throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");                                            \
		topkBlockRadixSortKernel<F, THREADS, WORK>                                                                                                   \
			<<<blockCount, exec.blockSize, sharedMemory>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.gridSize, in.k);                           \
		break

#define CASE_TopkBlockRadixSortKernel(THREADS)                                                                                                       \
	case THREADS: {                                                                                                                                  \
		std::uint32_t radixThreadWork = (in.k + THREADS - 1) / THREADS;                                                                              \
		switch (radixThreadWork) {                                                                                                                   \
			CASE_TopkBlockRadixSortKernelInner(THREADS, 1);                                                                                          \
			CASE_TopkBlockRadixSortKernelInner(THREADS, 2);                                                                                          \
			CASE_TopkBlockRadixSortKernelInner(THREADS, 3);                                                                                          \
			CASE_TopkBlockRadixSortKernelInner(THREADS, 4);                                                                                          \
			CASE_TopkBlockRadixSortKernelInner(THREADS, 5);                                                                                          \
			CASE_TopkBlockRadixSortKernelInner(THREADS, 6);                                                                                          \
			default:                                                                                                                                 \
				throw std::runtime_error("K param is too big.");                                                                                     \
		}                                                                                                                                            \
		break;                                                                                                                                       \
	}

	// unfortunatelly, if we stick with CUB radix sort we need to provide blockDim as template parameter
	switch (exec.blockSize) {
		CASE_TopkBlockRadixSortKernel(8);
		CASE_TopkBlockRadixSortKernel(16);
		CASE_TopkBlockRadixSortKernel(32);
		CASE_TopkBlockRadixSortKernel(64);
		CASE_TopkBlockRadixSortKernel(128);
		CASE_TopkBlockRadixSortKernel(256);
		CASE_TopkBlockRadixSortKernel(512);
		CASE_TopkBlockRadixSortKernel(1024);
		default:
			throw std::runtime_error("Unsupported block size.");
	}
}


/**
 * Each 2D block (X,Y) caches Y points to the shared memory (there are n/Y blocks in total)
 * and it iterates over grid points by caching batches of size X*itemsPerThreadX to the shared memory every step.
 * Each step, block threads fill [X*itemsPerThreadX,Y] matrix of distances. 
 * Then in parallel, Y threads perform serial insertion sort to the topK buffer in shared memory.
 * When iteration over grid points finishes, buffered topKs are written to the result array.
 */
template <typename F>
__global__ void topk2DBlockInsertionSortKernel(const F* const __restrict__ points, const F* const __restrict__ grid,
											   typename TopkProblemInstance<F>::Result* const __restrict__ topKs, const std::uint32_t dim,
											   const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const std::uint32_t itemsPerThreadX)
{
	extern __shared__ char sharedMemory[];

	F* const pointsCache = reinterpret_cast<F*>(sharedMemory);

	F* const gridPointsCache = pointsCache + blockDim.y * dim;
	const std::uint32_t gridPointsLeadingDim = dim + ((dim * sizeof(F)) % 32 == 0 ? 1 : 0); // if each gridPoint address was in same 32b bank
																							// we would have blockDim.x-way conflics

	F* const mergingCache = gridPointsCache + blockDim.x * gridPointsLeadingDim * itemsPerThreadX;
	const std::uint32_t mergingCacheLeadingDim = blockDim.x * itemsPerThreadX;
	typename TopkProblemInstance<F>::Result* const topKCache =
		reinterpret_cast<typename TopkProblemInstance<F>::Result*>(mergingCache + mergingCacheLeadingDim * blockDim.y);

	const std::uint32_t pointLimit = min(blockDim.y, n - (blockIdx.x * blockDim.y));
	const std::uint32_t planeIdx = threadIdx.x + threadIdx.y * blockDim.x;

	const auto tile = cg::tiled_partition(cg::this_thread_block(), blockDim.x);

	// initialize shared memory
	{
		// store points into shared memory
		// they will stay relevant throughout the whole block lifetime
		if (planeIdx < pointLimit)
			memcpy(pointsCache + planeIdx * dim, points + (blockIdx.x * blockDim.y + planeIdx) * dim, dim * sizeof(F));

		// clear topKCache
		for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x)
			topKCache[threadIdx.y * k + i].distance = valueMax<F>;

		__syncthreads();
	}

	for (std::uint32_t gridPointsOffset = 0; gridPointsOffset < gridSize; gridPointsOffset += mergingCacheLeadingDim) 
	{
		const std::uint32_t gridLimit = min(mergingCacheLeadingDim, gridSize - gridPointsOffset);

		// load grid points into shared memory
		{
			for (std::uint32_t gridIdx = planeIdx; gridIdx < gridLimit; gridIdx += blockDim.x + blockDim.y) 
				memcpy(gridPointsCache + gridIdx * gridPointsLeadingDim, grid + (gridPointsOffset + gridIdx) * dim, dim * sizeof(F));

			__syncthreads();
		}

		// compute distances for each thread in the block and store it in merging shared memory
		{
			for (std::uint32_t work = 0; work < itemsPerThreadX; ++work) 
			{
				F localDistance;
				const std::uint32_t gridIdx = threadIdx.x * itemsPerThreadX + work;

				if (gridIdx < gridLimit && threadIdx.y < pointLimit)
					localDistance = distance<F>(pointsCache + threadIdx.y * dim, gridPointsCache + gridIdx * gridPointsLeadingDim, dim);
				else
					localDistance = valueMax<F>;

				mergingCache[threadIdx.y * mergingCacheLeadingDim + gridIdx] = localDistance;
			}

			tile.sync();
		}

		// merge mergingCache into topKCache using serial insertion sort
		{
			if (threadIdx.x == 0 && threadIdx.y < pointLimit) { 
				const std::uint32_t localTopKCacheOffset = threadIdx.y * k;

				for (std::uint32_t i = 0; i < mergingCacheLeadingDim; ++i) {
					const F dist = mergingCache[threadIdx.y * mergingCacheLeadingDim + i];

					if (topKCache[localTopKCacheOffset + k - 1].distance > dist) {
						topKCache[localTopKCacheOffset + k - 1] = { dist, gridPointsOffset + i };
						bubbleUp<F>(topKCache + localTopKCacheOffset, k - 1);
					}
				}
			}
		}

		__syncthreads();
	}

	// finally, write topKCache to global memory
	{
		if (threadIdx.y < pointLimit) {
			for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x)
				topKs[(blockIdx.x * blockDim.y + threadIdx.y) * k + i] = topKCache[threadIdx.y * k + i];
		}
	}
}

// runner wrapped in a class
template <typename F>
void Topk2DBlockInsertionSortKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	const unsigned int threadsPerPoint = exec.blockSize / exec.itemsPerBlock;
	const unsigned int blockCount = (in.n + exec.itemsPerBlock - 1) / exec.itemsPerBlock;
	const unsigned sharedMemory = exec.itemsPerBlock * in.dim * sizeof(F) +									   // for points
								  exec.itemsPerThread * threadsPerPoint * (in.dim + 1) * sizeof(F) +		   // for grid (+1 for possible alignment)
								  exec.itemsPerThread * exec.blockSize * sizeof(F) +						   // for merging matrix
								  exec.itemsPerBlock * in.k * sizeof(typename TopkProblemInstance<F>::Result); // for topK cache

	if (sharedMemory > exec.sharedMemorySize)
		throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");

	topk2DBlockInsertionSortKernel<F><<<blockCount, dim3(threadsPerPoint, exec.itemsPerBlock), sharedMemory>>>(
		in.points, in.grid, in.topKs, in.dim, in.n, in.gridSize, in.k, exec.itemsPerThread);
}


/**
 * Uses bitonic top-k selection (modified bitonic sort). No inputs are explicitly cached, shm is used for intermediate topk results
 * BLOCK_SIZE - number of threads working on one point
 */
template <typename F, int GRID_SIZE>
__global__ void topkBitonicSortKernel(const F* __restrict__ points, const F* const __restrict__ grid, typename TopkProblemInstance<F>::Result* __restrict__ topKs, const std::uint32_t dim, const std::uint32_t n,
									  const std::uint32_t k)
{
	extern __shared__ char sharedMemory[];
	typename TopkProblemInstance<F>::Result* const shmTopk = (typename TopkProblemInstance<F>::Result*)(sharedMemory);
	typename TopkProblemInstance<F>::Result* const shmTopkBlock = shmTopk + GRID_SIZE * (threadIdx.x / (GRID_SIZE / 2)); // topk chunk that belongs to this thread

	// assign correct point and topK pointers for a thread (K/2 threads cooperate on each point)
	const std::uint32_t pointIdx = (blockIdx.x * blockDim.x + threadIdx.x) / (GRID_SIZE / 2);

	if (pointIdx >= n)
		return;

	points += pointIdx * dim;
	topKs += pointIdx * k;

	// compute all distances
	for (std::uint32_t i = threadIdx.x % (GRID_SIZE/2); i < GRID_SIZE; i += (GRID_SIZE / 2)) { // yes, this loop should go off exactly twice
		shmTopkBlock[i] = { distance<F>(points, grid + dim * i, dim), i };
	}
	__syncthreads();

	// final sorting of the topk result
	bitonic_sort<typename TopkProblemInstance<F>::Result, GRID_SIZE / 2>(shmTopk);
	__syncthreads();

	// copy topk results from shm to global memory
	{
		//topKs += blockIdx.x * blockDim.x * k / (GRID_SIZE / 2) + k * threadIdx.x / (GRID_SIZE / 2);
		unsigned i = threadIdx.x % (GRID_SIZE / 2);
		if (i < k) { // only first k items are copied
			topKs[i] = shmTopkBlock[i];
		}
	}
}

// runner wrapped in a class
template <typename F>
void TopkBitonicSortKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	if (exec.blockSize * 2 != (exec.blockSize | (exec.blockSize - 1)) + 1) {
		throw std::runtime_error("CUDA block size must be a power of two for bitonic topk selection.");
	}
	if (in.gridSize / 2 > exec.blockSize) {
		throw std::runtime_error("CUDA block size must be at least k/2.");
	}

	unsigned int blockCount = ((in.n * in.gridSize / 2) + exec.blockSize - 1) / exec.blockSize;
	unsigned int shmSize = exec.blockSize * 2 * sizeof(typename TopkProblemInstance<F>::Result); // 2 items per thread
	switch (in.gridSize) {
		case 2:
			topkBitonicSortKernel<F, 2><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 4:
			topkBitonicSortKernel<F, 4><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 8:
			topkBitonicSortKernel<F, 8><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 16:
			topkBitonicSortKernel<F, 16><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 32:
			topkBitonicSortKernel<F, 32><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 64:
			topkBitonicSortKernel<F, 64><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 128:
			topkBitonicSortKernel<F, 128><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 256:
			topkBitonicSortKernel<F, 256><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 512:
			topkBitonicSortKernel<F, 256><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		case 1024:
			topkBitonicSortKernel<F, 256><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.k);
			return;
		default:
			throw std::runtime_error("Given gridSize must be a power of two in [2,1024] range.");
	}
}



/**
 * Uses bitonic top-k selection (modified bitonic sort). No inputs are explicitly cached, shm is used for intermediate topk results 
 * @tparam K is number of top-k selected items and number of threads working cooperatively (keeping 2xK intermediate result in shm)
 */
template <typename F, int K>
__global__ void topkBitonicKernel(const F* __restrict__ points, const F* const __restrict__ grid, typename TopkProblemInstance<F>::Result* __restrict__ topKs, const std::uint32_t dim, const std::uint32_t n,
								  const std::uint32_t gridSize, const std::uint32_t actualK)
{
	extern __shared__ char sharedMemory[];
	typename TopkProblemInstance<F>::Result* const shmTopk = (typename TopkProblemInstance<F>::Result*)(sharedMemory);
	typename TopkProblemInstance<F>::Result* const shmTopkBlock = shmTopk + 2*K * (threadIdx.x / K); // topk chunk that belongs to this thread
	typename TopkProblemInstance<F>::Result* const shmNewData = shmTopk + (blockDim.x * 2); // every thread works on two items at a time 
	typename TopkProblemInstance<F>::Result* const shmNewDataBlock = shmNewData + 2*K * (threadIdx.x / K); // newData chunk that belongs to this thread

	// assign correct point and topK pointers for a thread (K/2 threads cooperate on each point)
	{
		const std::uint32_t pointIdx = (blockIdx.x * blockDim.x + threadIdx.x) / K;

		if (pointIdx >= n)
			return;

		points += pointIdx * dim;
		topKs += pointIdx * actualK;
	}

	// fill in initial topk intermediate result
	for (std::uint32_t i = threadIdx.x % K; i < 2*K; i += K) { // yes, this loop should go off exactly twice
		shmTopkBlock[i] = { distance<F>(points, grid + dim * i, dim), i };
	}

	// process the grid points in K-sized blocks
	for (std::uint32_t gridOffset = 2*K; gridOffset < gridSize; gridOffset += 2*K) {
		// compute another K new distances
		for (std::uint32_t i = threadIdx.x % K; i < 2*K; i += K) { // yes, this loop should go off exactly twice
			shmNewDataBlock[i] = { distance<F>(points, grid + dim * (gridOffset + i), dim), gridOffset + i };
		}

		__syncthreads(); // actually, whole block should be synced as the bitonic update operates on the whole block

		// merge them with intermediate topk
		bitonic_topk_update<typename TopkProblemInstance<F>::Result, K>(shmTopk, shmNewData);

		__syncthreads();
	}

	// final sorting of the topk result
	bitonic_sort<typename TopkProblemInstance<F>::Result, K>(shmTopk);
	__syncthreads();

	// copy topk results from shm to global memory
	if (threadIdx.x % K < actualK) {
		topKs[threadIdx.x % K] = shmTopkBlock[threadIdx.x % K];
	}
}


template <typename F, int K = 2>
void runnerWrapperBitonic(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	if constexpr (K > 256) {
		// a fallback (better run something slowly, than nothing at all)
		TopkBaseKernel<F>::run(in, exec);
	}
	else if (K < 2 || K < in.k) {
		// still looking for the right K...
		runnerWrapperBitonic<F, K * 2>(in, exec);
	}
	else {
		// we found the right nearest power of two using template meta-programming
		if (exec.blockSize * 2 != (exec.blockSize | (exec.blockSize - 1)) + 1) {
			throw std::runtime_error("CUDA block size must be a power of two for bitonic topk selection.");
		}
		if (K > exec.blockSize) {
			throw std::runtime_error("CUDA block size must be at least k.");
		}

		unsigned int blockCount = ((in.n * K) + exec.blockSize - 1) / exec.blockSize;
		unsigned int shmSize =
			exec.blockSize * 4 * sizeof(typename TopkProblemInstance<F>::Result); // 2 items per thread in topk and 2 more in tmp for new data

		topkBitonicKernel<F, K><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.gridSize, in.k);
	}
}


// runner wrapped in a class
template <typename F>
void TopkBitonicKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	runnerWrapperBitonic<F, 2>(in, exec);
}



/**
 * Uses bitonic top-k selection (modified bitonic sort). No inputs are explicitly cached, shm is used for intermediate topk results
 * @tparam K is number of top-k selected items and number of threads working cooperatively (keeping 2xK intermediate result in shm)
 * note that K is a power of 2 which is the nearest greater or equal value to actualK
 */
template <typename F, int K>
__global__ void topkBitonicOptKernel(const F* __restrict__ points, const F* const __restrict__ grid,
								  typename TopkProblemInstance<F>::Result* __restrict__ topKs, const std::uint32_t dim, const std::uint32_t n,
								  const std::uint32_t gridSize, const std::uint32_t actualK)
{
	extern __shared__ char sharedMemory[];
	typename TopkProblemInstance<F>::Result* const shmTopk = (typename TopkProblemInstance<F>::Result*)(sharedMemory);
	typename TopkProblemInstance<F>::Result* const shmTopkBlock = shmTopk + K * (threadIdx.x / (K/2)); // topk chunk that belongs to this thread
	typename TopkProblemInstance<F>::Result* const shmNewData = shmTopk + (blockDim.x * 2);			   // every thread works on two items at a time
	typename TopkProblemInstance<F>::Result* const shmNewDataBlock = shmNewData + K * (threadIdx.x / (K / 2)); // newData chunk that belongs to this thread

	// assign correct point and topK pointers for a thread (K/2 threads cooperate on each point)
	{
		const std::uint32_t pointIdx = (blockIdx.x * blockDim.x + threadIdx.x) / (K / 2);

		if (pointIdx >= n)
			return;

		points += pointIdx * dim;
		topKs += pointIdx * actualK;
	}

	// fill in initial topk intermediate result
	for (std::uint32_t i = threadIdx.x % (K / 2); i < K; i += (K / 2)) { // yes, this loop should go off exactly twice
		shmTopkBlock[i] = { distance<F>(points, grid + dim * i, dim), i };
	}

	// process the grid points in K-sized blocks
	for (std::uint32_t gridOffset = K; gridOffset < gridSize; gridOffset += K) {
		// compute another K new distances
		for (std::uint32_t i = threadIdx.x % (K / 2); i < K; i += (K / 2)) { // yes, this loop should go off exactly twice
			shmNewDataBlock[i] = { distance<F>(points, grid + dim * (gridOffset + i), dim), gridOffset + i };
		}

		__syncthreads(); // actually, whole block should be synced as the bitonic update operates on the whole block

		// merge them with intermediate topk
		bitonic_topk_update_opt<typename TopkProblemInstance<F>::Result, K / 2>(shmTopk, shmNewData);

		__syncthreads();
	}

	// final sorting of the topk result
	bitonic_sort<typename TopkProblemInstance<F>::Result, K / 2>(shmTopk);
	__syncthreads();

	// copy topk results from shm to global memory
	for (std::uint32_t i = threadIdx.x % (K / 2); i < actualK; i += (K / 2)) { // note there is actual K as limit (which might not be power of 2)
		topKs[i] = shmTopkBlock[i];
	}
}

template<typename F, int K = 2>
void runnerWrapperBitonicOpt(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	if constexpr (K > 256) {
		// a fallback (better run something slowly, than nothing at all)
		TopkBaseKernel<F>::run(in, exec);
	}
	else if (K < 2 || K < in.k) {
		// still looking for the right K...
		runnerWrapperBitonicOpt<F, K * 2>(in, exec);
	}
	else {
		// we found the right nearest power of two using template meta-programming
		if (exec.blockSize * 2 != (exec.blockSize | (exec.blockSize - 1)) + 1) {
			throw std::runtime_error("CUDA block size must be a power of two for bitonic topk selection.");
		}
		if (K / 2 > exec.blockSize) {
			throw std::runtime_error("CUDA block size must be at least half of k (rounded u to nearest power of 2).");
		}

		unsigned int blockCount = ((in.n * K / 2) + exec.blockSize - 1) / exec.blockSize;
		unsigned int shmSize =
			exec.blockSize * 4 * sizeof(typename TopkProblemInstance<F>::Result); // 2 items per thread in topk and 2 more in tmp for new data

		topkBitonicOptKernel<F, K><<<blockCount, exec.blockSize, shmSize>>>(in.points, in.grid, in.topKs, in.dim, in.n, in.gridSize, in.k);
	}
}

// runner wrapped in a class
template <typename F>
void TopkBitonicOptKernel<F>::run(const TopkProblemInstance<F>& in, CudaExecParameters& exec)
{
	runnerWrapperBitonicOpt<F, 2>(in, exec);
}



/*
 * Explicit template instantiation.
 */
template <typename F>
void instantiateKernelRunnerTemplates()
{
	TopkProblemInstance<F> instance(nullptr, nullptr, nullptr, 0, 0, 0, 0);
	CudaExecParameters exec;

	TopkBaseKernel<F>::run(instance, exec);
	TopkThreadSharedKernel<F>::run(instance, exec);
	TopkBlockRadixSortKernel<F>::run(instance, exec);
	Topk2DBlockInsertionSortKernel<F>::run(instance, exec);
	TopkBitonicSortKernel<F>::run(instance, exec);
	TopkBitonicKernel<F>::run(instance, exec);
	TopkBitonicOptKernel<F>::run(instance, exec);
}

template void instantiateKernelRunnerTemplates<float>();
#ifndef NO_DOUBLES
template void instantiateKernelRunnerTemplates<double>();
#endif
