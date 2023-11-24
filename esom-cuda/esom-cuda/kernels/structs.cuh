#ifndef ESOM_CUDA_STRUCTS_CUH
#define ESOM_CUDA_STRUCTS_CUH

#include <cstdint>

// macro-ized attributes for functions which need to be compiled both for GPU (nvcc) and for host (gcc/mvcc)
#ifdef __CUDACC__
#	define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#	define CUDA_CALLABLE_MEMBER
#endif

/**
 * Common CUDA kernel execution parameters. Each kernel runner may interpret them slightly differently,
 * but blockSize and sharedMemorySize are usually taken literally.
 */
struct CudaExecParameters
{
public:
	unsigned int blockSize;
	unsigned int sharedMemorySize;
	std::uint32_t itemsPerBlock;	// affects data replication/privatization (if the kernel employs it)
	std::uint32_t itemsPerThread;	// affects workload division among the threads
	std::uint32_t regsCache;		// affects size of the data cached in registers
	std::uint32_t groupsPerBlock;	// affects the number of thread groups in the block

	CudaExecParameters(unsigned int _blockSize = 256, unsigned int _sharedMemorySize = 0, std::uint32_t _itemsPerBlock = 1,
					   std::uint32_t _itemsPerThread = 1, std::uint32_t _regsCache = 1, std::uint32_t _groupsPerBlock = 1)
		: blockSize(_blockSize),
		  sharedMemorySize(_sharedMemorySize),
		  itemsPerBlock(_itemsPerBlock),
		  itemsPerThread(_itemsPerThread),
		  regsCache(_regsCache),
		  groupsPerBlock(_groupsPerBlock)
	{}
};


/**
 * Structure that aggregates all data (parameters, input buffers, output buffers) required for kernel execution.
 */
template<typename F = float>
struct TopkProblemInstance
{
public:
	struct Result
	{
		F distance;
		std::uint32_t index;

		CUDA_CALLABLE_MEMBER bool operator<(const Result& rhs) const
		{
			return this->distance < rhs.distance || (this->distance == rhs.distance && this->index < rhs.index);
		}
		
		CUDA_CALLABLE_MEMBER bool operator>(const Result& rhs) const
		{
			return rhs < *this;
		}
		
		CUDA_CALLABLE_MEMBER bool operator<=(const Result& rhs) const
		{
			return !(*this > rhs);
		}
		
		CUDA_CALLABLE_MEMBER bool operator>=(const Result& rhs) const
		{
			return !(*this < rhs);
		}
	};

	const F* points;
	const F* grid;
	Result* topKs;
	std::uint32_t dim;
	std::uint32_t n;
	std::uint32_t gridSize;
	std::uint32_t k;

	TopkProblemInstance(const F* _points, const F* _grid, Result* _topKs, std::size_t _dim, std::size_t _n, std::size_t _gridSize, std::size_t _k)
		: points(_points),
		  grid(_grid),
		  topKs(_topKs),
		  dim((std::uint32_t)_dim),
		  n((std::uint32_t)_n),
		  gridSize((std::uint32_t)_gridSize),
		  k((std::uint32_t)_k)
	{}
};


/**
 * Structure that aggregates all data (parameters, input buffers, output buffers) required for kernel execution.
 */
template <typename F = float>
struct ProjectionProblemInstance
{
	using TopkResult = typename TopkProblemInstance<F>::Result;

public:
	static constexpr F minBoost = F(1e-5);
	static constexpr F maxAvoidance = F(10);
	static constexpr F zeroAvoidance = F(1e-10);
	static constexpr F gridGravity = F(1e-5);

	const F* points;
	const F* grid;
	const F* grid2d;
	TopkResult* neighbors;
	F* projections;
	std::uint32_t dim;
	std::uint32_t n;
	std::uint32_t gridSize;
	std::uint32_t k;
	F adjust;
	F boost;
	std::uint32_t gridCacheLeadingDim;

	ProjectionProblemInstance(const F* _points, const F* _grid, const F* _grid2d, TopkResult* _neighbors, F* _projections, std::size_t _dim,
							  std::size_t _n, std::size_t _gridSize, std::size_t _k, F _adjust, F _boost, std::size_t _gridCacheLeadingDim = 0)
		: points(_points),
		  grid(_grid),
		  grid2d(_grid2d),
		  neighbors(_neighbors),
		  projections(_projections),
		  dim((std::uint32_t)_dim),
		  n((std::uint32_t)_n),
		  gridSize((std::uint32_t)_gridSize),
		  k((std::uint32_t)_k),
		  adjust(_adjust),
		  boost(_boost),
		  gridCacheLeadingDim((std::uint32_t)_gridCacheLeadingDim)
	{}
};



#endif