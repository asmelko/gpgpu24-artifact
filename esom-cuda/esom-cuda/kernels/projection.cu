#ifdef __INTELLISENSE__
#	define __CUDACC__
#endif

#define _CG_ABI_EXPERIMENTAL

#include "cooperative_groups.h"
#include "cooperative_groups/memcpy_async.h"
#include "cooperative_groups/reduce.h"
#include "cub/block/block_reduce.cuh"
#include "cub/warp/warp_reduce.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "projection.cuh"

namespace cg = cooperative_groups;

template <typename F>
using TopkResult = typename TopkProblemInstance<F>::Result;

template <typename F>
struct ArrayFloatType
{};

template <>
struct ArrayFloatType<float>
{
	using Type2 = float2;
	using Type4 = float4;
};

template <>
struct ArrayFloatType<double>
{
	using Type2 = double2;
	using Type4 = double4;
};

template <typename F>
__inline__ __device__ void sortedDistsToScores(TopkResult<F>* const __restrict__ neighbors, const std::size_t adjustedK, const std::size_t k,
											   const F boost)
{
	// compute the distance distribution for the scores
	F mean = 0, sd = 0, wsum = 0;
	for (std::uint32_t i = 0; i < adjustedK; ++i) {
		const F tmp = sqrt(neighbors[i].distance);
		const F w = 1 / F(i + 1);
		mean += tmp * w;
		wsum += w;
		neighbors[i].distance = tmp;
	}

	mean /= wsum;

	for (std::uint32_t i = 0; i < adjustedK; ++i) {
		const F tmp = neighbors[i].distance - mean;
		const F w = 1 / F(i + 1);
		sd += tmp * tmp * w;
	}

	sd = boost / sqrt(sd / wsum);
	const F nmax = ProjectionProblemInstance<F>::maxAvoidance / neighbors[adjustedK - 1].distance;

	// convert the stuff to scores
	for (std::uint32_t i = 0; i < k; ++i) {
		if (k < adjustedK)
			neighbors[i].distance =
				exp((mean - neighbors[i].distance) * sd) * (1 - exp(neighbors[i].distance * nmax - ProjectionProblemInstance<F>::maxAvoidance));
		else
			neighbors[i].distance = exp((mean - neighbors[i].distance) * sd);
	}
}

/**
 * Helper structure to help transform neighbor distances to scores.
 * Shares `distance` field for storing scores and loading distances.
 */
template <typename F>
struct SharedNeighborStorage
{
	TopkResult<F>* const __restrict__ neighbors;

	__forceinline__ __device__ F getNeighborDistance(const std::uint32_t idx) const
	{
		return neighbors[idx].distance;
	}

	__forceinline__ __device__ void storeScore(const std::uint32_t idx, const F score)
	{
		neighbors[idx].distance = score;
	}
};

/**
 * Helper structure to help transform neighbor distances to scores.
 * Separate arrays for distances and scores.
 */
template <typename F>
struct NeighborScoreStorage
{
	const TopkResult<F>* const __restrict__ neighbors;
	F* const __restrict__ scores;

	__forceinline__ __device__ F getNeighborDistance(const std::uint32_t idx) const
	{
		return neighbors[idx].distance;
	}

	__forceinline__ __device__ void storeScore(const std::uint32_t idx, const F score)
	{
		scores[idx] = score;
	}
};

/**
 * Uses CUB for reduction and shuffle.
 */
template <typename F, class SCORE_STORAGE>
__inline__ __device__ void sortedDistsToScoresGroup(SCORE_STORAGE storage, char* const __restrict__ sharedMemory, const std::size_t adjustedK,
												   const std::size_t k, const F boost)
{
	typedef cub::WarpReduce<F> WarpReduce;
	auto* const reduceStorage = reinterpret_cast<typename WarpReduce::TempStorage*>(sharedMemory);

	// each thread in warp can have at most 3 scores as k <= 64 (adjustedK <= k + 1)
	F tmpScores[3];
	F lastScore;

	// compute the distance distribution for the scores
	F mean = 0, sd = 0, wsum = 0;
	for (std::uint32_t i = threadIdx.x; i < adjustedK; i += warpSize) {
		const F tmp = sqrt(storage.getNeighborDistance(i));
		const F w = 1 / F(i + 1);
		mean += tmp * w;
		wsum += w;
		tmpScores[i / warpSize] = tmp;
	}

	{
		mean = WarpReduce(*reduceStorage).Sum(mean);
		mean = cub::ShuffleIndex<32>(mean, 0, 0xffffffff);

		wsum = WarpReduce(*reduceStorage).Sum(wsum);
		wsum = cub::ShuffleIndex<32>(wsum, 0, 0xffffffff);
	}

	mean /= wsum;

	for (std::uint32_t i = threadIdx.x; i < adjustedK; i += warpSize) {
		const F tmp = tmpScores[i / warpSize] - mean;
		const F w = 1 / F(i + 1);
		sd += tmp * tmp * w;
	}

	{
		sd = WarpReduce(*reduceStorage).Sum(sd);
		sd = cub::ShuffleIndex<32>(sd, 0, 0xffffffff);

		const auto lastScoreThreadIdx = (adjustedK - 1) % warpSize;
		const auto lastScoreIdx = (adjustedK - 1) / warpSize;
		lastScore = cub::ShuffleIndex<32>(tmpScores[lastScoreIdx], lastScoreThreadIdx, 0xffffffff);
	}

	sd = boost / sqrt(sd / wsum);
	const F nmax = ProjectionProblemInstance<F>::maxAvoidance / lastScore;

	// convert the stuff to scores
	if (k < adjustedK)
		for (std::uint32_t i = threadIdx.x; i < k; i += warpSize) {
			const auto scoreIdx = i / warpSize;
			const F score =
				exp((mean - tmpScores[scoreIdx]) * sd) * (1 - exp(tmpScores[scoreIdx] * nmax - ProjectionProblemInstance<F>::maxAvoidance));
			storage.storeScore(i, score);
		}
	else
		for (std::uint32_t i = threadIdx.x; i < k; i += warpSize)
			storage.storeScore(i, exp((mean - tmpScores[i / warpSize]) * sd));
}

/**
 * Uses thread_block_tile and its built in functions for reduction and shuffle.
 */
template <typename F, class SCORE_STORAGE, class TILE>
__inline__ __device__ void sortedDistsToScoresGroup(const TILE& tile, SCORE_STORAGE storage, const std::size_t adjustedK, const std::size_t k,
												   const F boost)
{
	// if k is big enough and tile is small enough, this array can overflow... should be MAX_K / tile.size()
	F tmpScores[10];
	F lastScore;

	// compute the distance distribution for the scores
	F mean = 0, sd = 0, wsum = 0;
	for (std::uint32_t i = tile.thread_rank(); i < adjustedK; i += tile.size()) {
		const F tmp = sqrt(storage.getNeighborDistance(i));
		const F w = 1 / F(i + 1);
		mean += tmp * w;
		wsum += w;
		tmpScores[i / tile.size()] = tmp;
	}

	{
		mean = cg::reduce(tile, mean, cg::plus<F>());
		wsum = cg::reduce(tile, wsum, cg::plus<F>());
	}

	mean /= wsum;

	for (std::uint32_t i = tile.thread_rank(); i < adjustedK; i += tile.size()) {
		const F tmp = tmpScores[i / tile.size()] - mean;
		const F w = 1 / F(i + 1);
		sd += tmp * tmp * w;
	}

	{
		sd = cg::reduce(tile, sd, cg::plus<F>());

		const auto lastScoreThreadIdx = (adjustedK - 1) % tile.size();
		const auto lastScoreIdx = (adjustedK - 1) / tile.size();
		lastScore = tile.shfl(tmpScores[lastScoreIdx], lastScoreThreadIdx);
	}

	sd = boost / sqrt(sd / wsum);
	const F nmax = ProjectionProblemInstance<F>::maxAvoidance / lastScore;

	// convert the stuff to scores
	if (k < adjustedK)
		for (std::uint32_t i = tile.thread_rank(); i < k; i += tile.size()) {
			const auto scoreIdx = i / tile.size();
			const F score =
				exp((mean - tmpScores[scoreIdx]) * sd) * (1 - exp(tmpScores[scoreIdx] * nmax - ProjectionProblemInstance<F>::maxAvoidance));
			storage.storeScore(i, score);
		}
	else
		for (std::uint32_t i = tile.thread_rank(); i < k; i += tile.size())
			storage.storeScore(i, exp((mean - tmpScores[i / tile.size()]) * sd));
}

template <typename F>
__inline__ __device__ void addGravity(const F score, const F* const __restrict__ grid2DPoint, F* const __restrict__ mtx)
{
	const F gs = score * ProjectionProblemInstance<F>::gridGravity;

	mtx[0] += gs;
	mtx[3] += gs;
	mtx[4] += gs * grid2DPoint[0];
	mtx[5] += gs * grid2DPoint[1];
}

template <typename F>
__inline__ __device__ void addGravity2Wise(const F score, const F* const __restrict__ grid2DPoint, F* const __restrict__ mtx)
{
	const F gs = score * ProjectionProblemInstance<F>::gridGravity;

	const typename ArrayFloatType<F>::Type2 tmpGrid2d = reinterpret_cast<const typename ArrayFloatType<F>::Type2*>(grid2DPoint)[0];

	mtx[0] += gs;
	mtx[3] += gs;
	mtx[4] += gs * tmpGrid2d.x;
	mtx[5] += gs * tmpGrid2d.y;
}

template <typename F>
__inline__ __device__ typename ArrayFloatType<F>::Type2 euclideanProjection(const F* const __restrict__ point, const F* const __restrict__ gridPointI,
																			const F* const __restrict__ gridPointJ, const std::uint32_t dim)
{
	typename ArrayFloatType<F>::Type2 result { 0.0, 0.0 };
	for (std::uint32_t k = 0; k < dim; ++k) {
		const F tmp = gridPointJ[k] - gridPointI[k];
		result.y += tmp * tmp;
		result.x += tmp * (point[k] - gridPointI[k]);
	}
	return result;
}


template <typename F>
__inline__ __device__ typename ArrayFloatType<F>::Type2 euclideanProjection4Wise(const F* const __restrict__ point,
																				 const F* const __restrict__ gridPointI,
																				 const F* const __restrict__ gridPointJ, const std::uint32_t dim)
{
	const auto* const __restrict__ gridPointI4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointI);
	const auto* const __restrict__ gridPointJ4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointJ);
	const auto* const __restrict__ point4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(point);

	typename ArrayFloatType<F>::Type2 result { 0.0, 0.0 };

#define DOIT(X)                                                                                                                                      \
	tmp = tmpGridJ.X - tmpGridI.X;                                                                                                                   \
	result.y += tmp * tmp;                                                                                                                           \
	result.x += tmp * (tmpPoint.X - tmpGridI.X)

	for (std::uint32_t k = 0; k < dim / 4; ++k) {
		const auto tmpGridI = gridPointI4[k];
		const auto tmpGridJ = gridPointJ4[k];
		const auto tmpPoint = point4[k];

		F tmp;
		DOIT(x);
		DOIT(y);
		DOIT(z);
		DOIT(w);
	}

	for (std::uint32_t k = dim - (dim % 4); k < dim; ++k) {
		const F tmp = gridPointJ[k] - gridPointI[k];
		result.y += tmp * tmp;
		result.x += tmp * (point[k] - gridPointI[k]);
	}

	return result;
}

template <typename F>
__inline__ __device__ void euclideanProjection4WiseRegister(
	const F* const __restrict__ point, const F* const __restrict__ gridPointI, const F* const __restrict__ gridPointJ, const F* const __restrict__ gridPointK,
															const F* const __restrict__ gridPointL, const std::uint32_t dim,
															typename ArrayFloatType<F>::Type2* const __restrict__ result)
{
	const auto* const __restrict__ gridPointI4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointI);
	const auto* const __restrict__ gridPointJ4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointJ);
	const auto* const __restrict__ gridPointK4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointK);
	const auto* const __restrict__ gridPointL4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointL);
	const auto* const __restrict__ point4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(point);

#define DOITREGISTER4(I, J, X)                                                                                                                       \
	tmp = tmpGrid[J].X - tmpGrid[I].X;                                                                                                               \
	result[I * 2 + J - 2].y += tmp * tmp;                                                                                                            \
	result[I * 2 + J - 2].x += tmp * (tmpPoint.X - tmpGrid[I].X)

	for (std::uint32_t k = 0; k < dim / 4; ++k) {
		const typename ArrayFloatType<F>::Type4 tmpGrid[4] { gridPointI4[k], gridPointJ4[k], gridPointK4[k], gridPointL4[k] };
		const auto tmpPoint = point4[k];

		F tmp;
		#pragma unroll
		for (std::uint32_t i = 0; i < 2; i++) {
			#pragma unroll
			for (std::uint32_t j = 2; j < 4; j++) {
				DOITREGISTER4(i, j, x);
				DOITREGISTER4(i, j, y);
				DOITREGISTER4(i, j, z);
				DOITREGISTER4(i, j, w);
			}
		}
	}

	for (std::uint32_t k = dim - (dim % 4); k < dim; ++k) {
		const F* const __restrict__ gridPoint[] { gridPointI, gridPointJ, gridPointK, gridPointL };

		#pragma unroll
		for (std::uint32_t i = 0; i < 2; i++) {
			#pragma unroll
			for (std::uint32_t j = 2; j < 4; j++) {
				const F tmp = gridPoint[j][k] - gridPoint[i][k];
				result[i * 2 + j - 2].y += tmp * tmp;
				result[i * 2 + j - 2].x += tmp * (point[k] - gridPoint[i][k]);
			}
		}
	}
}

template <typename F>
__inline__ __device__ void euclideanProjection4WiseRegister(const F* const __restrict__ point, const F* const __restrict__ gridPointI,
															const F* const __restrict__ gridPointJ, const F* const __restrict__ gridPointK,
															const std::uint32_t dim, typename ArrayFloatType<F>::Type2* const __restrict__ result)
{
	const auto* const __restrict__ gridPointI4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointI);
	const auto* const __restrict__ gridPointJ4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointJ);
	const auto* const __restrict__ gridPointK4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(gridPointK);
	const auto* const __restrict__ point4 = reinterpret_cast<const typename ArrayFloatType<F>::Type4*>(point);

#define DOITREGISTER3(I, J, X)                                                                                                                       \
	tmp = tmpGrid[J].X - tmpGrid[I].X;                                                                                                               \
	result[I + J - 1].y += tmp * tmp;                                                                                                                \
	result[I + J - 1].x += tmp * (tmpPoint.X - tmpGrid[I].X)

	for (std::uint32_t k = 0; k < dim / 4; ++k) {
		const typename ArrayFloatType<F>::Type4 tmpGrid[] { gridPointI4[k], gridPointJ4[k], gridPointK4[k] };
		const auto tmpPoint = point4[k];

		F tmp;

		DOITREGISTER3(0, 1, x);
		DOITREGISTER3(0, 1, y);
		DOITREGISTER3(0, 1, z);
		DOITREGISTER3(0, 1, w);

		DOITREGISTER3(0, 2, x);
		DOITREGISTER3(0, 2, y);
		DOITREGISTER3(0, 2, z);
		DOITREGISTER3(0, 2, w);

		DOITREGISTER3(1, 2, x);
		DOITREGISTER3(1, 2, y);
		DOITREGISTER3(1, 2, z);
		DOITREGISTER3(1, 2, w);
	}

	for (std::uint32_t k = dim - (dim % 4); k < dim; ++k) {
		const F* const __restrict__ gridPoint[] { gridPointI, gridPointJ, gridPointK };
		F tmp;

		tmp = gridPoint[1][k] - gridPoint[0][k];
		result[0].y += tmp * tmp;
		result[0].x += tmp * (point[k] - gridPoint[0][k]);

		tmp = gridPoint[2][k] - gridPoint[0][k];
		result[1].y += tmp * tmp;
		result[1].x += tmp * (point[k] - gridPoint[0][k]);

		tmp = gridPoint[2][k] - gridPoint[1][k];
		result[2].y += tmp * tmp;
		result[2].x += tmp * (point[k] - gridPoint[1][k]);
	}
}

template <typename F>
__inline__ __device__ void addApproximation(const F scoreI, const F scoreJ, const F* const __restrict__ grid2DPointI,
											const F* const __restrict__ grid2DPointJ, const F adjust, const F scalarProjection,
											F* const __restrict__ mtx)
{
	F h[2], hp = 0;
	#pragma unroll
	for (std::uint32_t i = 0; i < 2; ++i) {
		h[i] = grid2DPointJ[i] - grid2DPointI[i];
		hp += h[i] * h[i];
	}

	if (hp < ProjectionProblemInstance<F>::zeroAvoidance)
		return;

	const F exponent = scalarProjection - .5;
	const F s = scoreI * scoreJ * pow(1 + hp, adjust) * exp(-exponent * exponent);
	const F sihp = s / hp;
	const F rhsc = s * (scalarProjection + (h[0] * grid2DPointI[0] + h[1] * grid2DPointI[1]) / hp);

	mtx[0] += h[0] * h[0] * sihp;
	mtx[1] += h[0] * h[1] * sihp;
	mtx[2] += h[1] * h[0] * sihp;
	mtx[3] += h[1] * h[1] * sihp;
	mtx[4] += h[0] * rhsc;
	mtx[5] += h[1] * rhsc;
}

template <typename F>
__inline__ __device__ void addApproximation2Wise(const F scoreI, const F scoreJ, const F* const __restrict__ grid2DPointI,
												 const F* const __restrict__ grid2DPointJ, const F adjust, const F scalarProjection,
												 F* const __restrict__ mtx)
{
	const typename ArrayFloatType<F>::Type2 tmpGrid2dI = reinterpret_cast<const typename ArrayFloatType<F>::Type2*>(grid2DPointI)[0];
	const typename ArrayFloatType<F>::Type2 tmpGrid2dJ = reinterpret_cast<const typename ArrayFloatType<F>::Type2*>(grid2DPointJ)[0];

	const F h[2] { tmpGrid2dJ.x - tmpGrid2dI.x, tmpGrid2dJ.y - tmpGrid2dI.y };
	const F hp = h[0] * h[0] + h[1] * h[1];

	if (hp < ProjectionProblemInstance<F>::zeroAvoidance)
		return;

	const F exponent = scalarProjection - .5;
	const F s = scoreI * scoreJ * pow(1 + hp, adjust) * exp(-exponent * exponent);
	const F sihp = s / hp;
	const F rhsc = s * (scalarProjection + (h[0] * tmpGrid2dI.x + h[1] * tmpGrid2dI.y) / hp);

	mtx[0] += h[0] * h[0] * sihp;
	mtx[1] += h[0] * h[1] * sihp;
	mtx[2] += h[1] * h[0] * sihp;
	mtx[3] += h[1] * h[1] * sihp;
	mtx[4] += h[0] * rhsc;
	mtx[5] += h[1] * rhsc;
}

/**
 * One thread computes embedding for one point.
 */
template <typename F>
__global__ void projectionBaseKernel(const F* __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
									 TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
									 const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost)
{
	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;
		const std::uint32_t pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

		if (pointIdx >= n)
			return;

		points = points + pointIdx * dim;
		neighbors = neighbors + pointIdx * adjustedK;
		projections = projections + pointIdx * 2;

		sortedDistsToScores<F>(neighbors, adjustedK, k, boost);
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = 0; i < k; ++i) {
		const std::uint32_t idxI = neighbors[i].index;
		const F scoreI = neighbors[i].distance;

		addGravity(scoreI, grid2d + idxI * 2, mtx);

		for (std::uint32_t j = i + 1; j < k; ++j) {
			const std::uint32_t idxJ = neighbors[j].index;
			const F scoreJ = neighbors[j].distance;

			const auto result = euclideanProjection<F>(points, grid + idxI * dim, grid + idxJ * dim, dim);
			F scalarProjection = result.x;
			const F squaredGridPointsDistance = result.y;

			if (squaredGridPointsDistance == F(0))
				continue;

			scalarProjection /= squaredGridPointsDistance;

			addApproximation(scoreI, scoreJ, grid2d + idxI * 2, grid2d + idxJ * 2, adjust, scalarProjection, mtx);
		}
	}

	// solve linear equation
	const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
	projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
	projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
}

// runner wrapped in a class
template <typename F>
void ProjectionBaseKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = (in.n + exec.blockSize - 1) / exec.blockSize;
	projectionBaseKernel<F><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize,
															in.k, in.adjust, in.boost);
}

template <typename INDEXER>
__inline__ __device__ uint2 getIndices(std::uint32_t plainIndex, std::uint32_t k)
{}

/**
 * Assigns consecutive indices to consecutive threads.
 * However, branch divergence occurs.
 * Thread index assignments:
 * k|	0	1	2	3	4
 * -+--------------------
 * 0|		0	1	2	3
 * 1|			4	5	6
 * 2|				7	8
 * 3|					9
 * 4|
 */
struct BaseIndexer
{};
template <>
__inline__ __device__ uint2 getIndices<BaseIndexer>(std::uint32_t plainIndex, std::uint32_t k)
{
	--k;
	uint2 indices { 0, 0 };
	while (plainIndex >= k) {
		++indices.x;
		plainIndex -= k--;
	}
	indices.y = plainIndex + 1 + indices.x;
	return indices;
}

/**
 * "Concatenates" 2 columns into one (1. and k-1., 2. and k-2., ...) and
 * creates indexing on top of k * k/2 rectangle.
 * No branch divergence.
 * Thread index assignments:
 * k|	0	1	2	3	4
 * -+--------------------
 * 0|		0	1	2	3
 * 1|			5	6	7
 * 2|				8	9
 * 3|					4
 * 4|
 */
struct RectangleIndexer
{};
template <>
__inline__ __device__ uint2 getIndices<RectangleIndexer>(std::uint32_t plainIndex, std::uint32_t k)
{
	uint2 indices;
	const std::uint32_t tempI = plainIndex / k;
	const std::uint32_t tempJ = plainIndex % k;
	const auto invertedI = k - 1 - tempI;
	indices.x = tempJ < invertedI ? tempI : invertedI - 1;
	indices.y = (tempJ < invertedI ? tempJ : tempJ - invertedI) + indices.x + 1;

	return indices;
}

/**
 * One block computes embedding for one point, using CUB block reduce for matrix reduction.
 */
template <typename F, typename INDEXER>
__global__ void projectionBlockKernel(const F* __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
									  TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
									  const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost,
									  const std::uint32_t itemsPerThread)
{
	typedef cub::BlockReduce<F, 1024> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		points = points + blockIdx.x * dim;
		neighbors = neighbors + blockIdx.x * adjustedK;
		projections = projections + blockIdx.x * 2;

		cg::thread_block_tile<32, cg::thread_block> tile32 = cg::tiled_partition<32>(cg::this_thread_block());

		if (threadIdx.x < 32)
			sortedDistsToScoresGroup<F>(tile32, SharedNeighborStorage<F> { neighbors }, adjustedK, k, boost);

		__syncthreads();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		const auto neighbor = neighbors[i];
		addGravity(neighbor.distance, grid2d + neighbor.index * 2, mtx);
	}

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = threadIdx.x * itemsPerThread; i < neighborPairs; i += blockDim.x * itemsPerThread) {
		for (size_t j = 0; j < itemsPerThread; ++j) {
			const auto threadIndex = i + j;
			if (threadIndex >= neighborPairs)
				continue;

			const auto indices = getIndices<INDEXER>(threadIndex, k);

			const std::uint32_t idxI = neighbors[indices.x].index;
			const std::uint32_t idxJ = neighbors[indices.y].index;
			const F scoreI = neighbors[indices.x].distance;
			const F scoreJ = neighbors[indices.y].distance;

			const auto result = euclideanProjection<F>(points, grid + idxI * dim, grid + idxJ * dim, dim);
			F scalarProjection = result.x;
			const F squaredGridPointsDistance = result.y;

			if (squaredGridPointsDistance == F(0))
				continue;

			scalarProjection /= squaredGridPointsDistance;

			addApproximation(scoreI, scoreJ, grid2d + idxI * 2, grid2d + idxJ * 2, adjust, scalarProjection, mtx);
		}
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = BlockReduce(temp_storage).Sum(mtx[i], blockDim.x);
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	projectionBlockKernel<F, BaseIndexer><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n,
																		  in.gridSize, in.k, in.adjust, in.boost, exec.itemsPerThread);
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockRectangleIndexKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	projectionBlockKernel<F, RectangleIndexer><<<blockCount, exec.blockSize>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim,
																			   in.n, in.gridSize, in.k, in.adjust, in.boost, exec.itemsPerThread);
}

/**
 * One block computes embedding for one point, using CUB block reduce for matrix reduction.
 * All data that is used in the computation is copied to shared memory.
 */
template <typename F>
__global__ void projectionBlockSharedKernel(const F* const __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
											TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
											const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost)
{
	extern __shared__ char sharedMemory[];
	typedef cub::BlockReduce<F, 1024> BlockReduce;
	auto* const __restrict__ reduceStorage = reinterpret_cast<typename BlockReduce::TempStorage*>(sharedMemory);
	F* const __restrict__ pointCache = reinterpret_cast<F*>(sharedMemory + sizeof(typename BlockReduce::TempStorage));
	F* const __restrict__ scoresCache = pointCache + dim;
	F* const __restrict__ grid2dCache = scoresCache + k;
	F* const __restrict__ gridCache = grid2dCache + 2 * k;

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		neighbors = neighbors + blockIdx.x * adjustedK;
		projections = projections + blockIdx.x * 2;

		int copyIdx = threadIdx.x;

		if (threadIdx.x < warpSize) {
			// silently assuming that CUB storage needed for warp reduction is smaller that the one needed for block reduction
			sortedDistsToScoresGroup<F>(NeighborScoreStorage<F> { neighbors, scoresCache }, sharedMemory, adjustedK, k, boost);
			copyIdx += blockDim.x;
		}

		// ugly copying to shared
		{
			for (; copyIdx < warpSize + dim; copyIdx += blockDim.x) {
				auto cacheIdx = copyIdx - warpSize;
				pointCache[cacheIdx] = points[blockIdx.x * dim + cacheIdx];
			}

			for (; copyIdx < warpSize + dim + k * 2; copyIdx += blockDim.x) {
				auto cacheIdx = copyIdx - warpSize - dim;
				grid2dCache[cacheIdx] = grid2d[neighbors[cacheIdx / 2].index * 2 + (cacheIdx % 2)];
			}

			for (; copyIdx < warpSize + dim + k * 2 + k * dim; copyIdx += blockDim.x) {
				auto cacheIdx = copyIdx - k * 2 - warpSize - dim;
				auto globIdx = cacheIdx / dim;
				auto globOff = cacheIdx % dim;
				gridCache[cacheIdx] = grid[neighbors[globIdx].index * dim + globOff];
			}
		}

		__syncthreads();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
		addGravity(scoresCache[i], grid2dCache + i * 2, mtx);
	}

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = threadIdx.x; i < neighborPairs; i += blockDim.x) {
		const auto indices = getIndices<RectangleIndexer>(i, k);

		const auto I = indices.x;
		const auto J = indices.y;
		const F scoreI = scoresCache[I];
		const F scoreJ = scoresCache[J];

		const auto result = euclideanProjection<F>(pointCache, gridCache + I * dim, gridCache + J * dim, dim);
		F scalarProjection = result.x;
		const F squaredGridPointsDistance = result.y;

		if (squaredGridPointsDistance == F(0))
			continue;

		scalarProjection /= squaredGridPointsDistance;

		addApproximation(scoreI, scoreJ, grid2dCache + I * 2, grid2dCache + J * 2, adjust, scalarProjection, mtx);
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = BlockReduce(*reduceStorage).Sum(mtx[i], blockDim.x);
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockSharedKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	unsigned int blockCount = in.n;
	unsigned int sharedMemory = sizeof(typename cub::BlockReduce<F, 1024>::TempStorage) + // for reduction
								in.dim * sizeof(F) +									  // for point
								in.k * sizeof(F) +										  // for scores
								in.k * 2 * sizeof(F) +									  // for grid2d points
								in.k * in.dim * sizeof(F);								  // for grid points

	if (sharedMemory > exec.sharedMemorySize)
		throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");

	projectionBlockSharedKernel<F><<<blockCount, exec.blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim,
																				 in.n, in.gridSize, in.k, in.adjust, in.boost);
}


/**
 * One block computes embedding for one point, using built in experimental tile block reduce for matrix reduction.
 */
template <typename F, typename INDEXER, uint32_t TILE_SIZE, typename TILE>
__inline__ __device__ void projectionBlockKernelInternal(const TILE& g, F* __restrict__ sharedMemory, const F* const __restrict__ points,
														 const F* const __restrict__ grid, const F* const __restrict__ grid2d,
														 TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
														 const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust,
														 const F boost)
{
	sharedMemory += g.meta_group_rank() * (dim + k * 2 + k + dim * k);
	F* const __restrict__ pointCache = sharedMemory;
	F* const __restrict__ grid2dCache = sharedMemory + dim;
	F* const __restrict__ scoreCache = sharedMemory + dim + k * 2;
	F* const __restrict__ gridCache = sharedMemory + dim + k * 3;

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		const auto workIdx = blockIdx.x * g.meta_group_size() + g.meta_group_rank();

		if (workIdx >= n)
			return;

		neighbors = neighbors + workIdx * adjustedK;
		projections = projections + workIdx * 2;

		if constexpr (TILE_SIZE <= 32) {
			sortedDistsToScoresGroup<F>(g, NeighborScoreStorage<F> { neighbors, scoreCache }, adjustedK, k, boost);

			std::uint32_t i = g.thread_rank();
			for (; i < dim; i += g.size()) {
				pointCache[i] = (points + workIdx * dim)[i];
			}

			for (; i < dim + k * 2; i += g.size()) {
				const auto cacheIdx = i - dim;
				grid2dCache[cacheIdx] = grid2d[neighbors[cacheIdx / 2].index * 2 + (cacheIdx % 2)];
			}

			for (; i < dim + k * 2 + k * dim; i += g.size()) {
				auto cacheIdx = i - k * 2 - dim;
				auto globIdx = cacheIdx / dim;
				auto globOff = cacheIdx % dim;
				gridCache[cacheIdx] = grid[neighbors[globIdx].index * dim + globOff];
			}
		}
		if constexpr (TILE_SIZE > 32) {
			auto warp = cg::tiled_partition<32>(g);

			if (warp.meta_group_rank() == 0)
				sortedDistsToScoresGroup<F>(warp, NeighborScoreStorage<F> { neighbors, scoreCache }, adjustedK, k, boost);
			else {
				std::uint32_t i = g.thread_rank() - warpSize;
				for (; i < dim; i += g.size()) {
					pointCache[i] = (points + workIdx * dim)[i];
				}

				for (; i < dim + k * 2; i += g.size() - warpSize) {
					const auto cacheIdx = i - dim;
					grid2dCache[cacheIdx] = grid2d[neighbors[cacheIdx / 2].index * 2 + (cacheIdx % 2)];
				}

				for (; i < dim + k * 2 + k * dim; i += g.size() - warpSize) {
					auto cacheIdx = i - k * 2 - dim;
					auto globIdx = cacheIdx / dim;
					auto globOff = cacheIdx % dim;
					gridCache[cacheIdx] = grid[neighbors[globIdx].index * dim + globOff];
				}
			}

			cg::sync(g);
		}
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = g.thread_rank(); i < neighborPairs; i += g.size()) {
		const auto indices = getIndices<INDEXER>(i, k);

		std::uint32_t I = indices.x;
		std::uint32_t J = indices.y;

		if (I + 1 == J)
			addGravity(scoreCache[I], grid2dCache + I * 2, mtx);

		const auto result = euclideanProjection<F>(pointCache, gridCache + I * dim, gridCache + J * dim, dim);
		F scalarProjection = result.x;
		const F squaredGridPointsDistance = result.y;

		if (squaredGridPointsDistance == F(0))
			continue;

		scalarProjection /= squaredGridPointsDistance;

		addApproximation(scoreCache[I], scoreCache[J], grid2dCache + I * 2, grid2dCache + J * 2, adjust, scalarProjection, mtx);
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = cg::reduce(g, mtx[i], cg::plus<F>());
	}

	if (g.thread_rank() == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

/**
 * One block computes embedding for multiple point, using experimenta built in tile reduce for matrix reduction.
 */
template <typename F, class INDEXER, std::uint32_t TILE_SIZE>
__global__ void projectionBlockMultiKernel(const F* __restrict__ points, const F* const __restrict__ grid, const F* const __restrict__ grid2d,
										   TopkResult<F>* __restrict__ neighbors, F* __restrict__ projections, const std::uint32_t dim,
										   const std::uint32_t n, const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost)
{
	extern __shared__ char sharedMem[];
	auto* __restrict__ block_mem = reinterpret_cast<cg::block_tile_memory<1024>*>(sharedMem);
	F* __restrict__ pointCache = reinterpret_cast<F*>(block_mem + 1);

	auto block = cg::this_thread_block(*block_mem);
	auto tile = cg::tiled_partition<TILE_SIZE>(block);

	projectionBlockKernelInternal<F, INDEXER, TILE_SIZE>(tile, pointCache, points, grid, grid2d, neighbors, projections, dim, n, gridSize, k, adjust,
														 boost);
}

// runner wrapped in a class
template <typename F>
void ProjectionBlockMultiKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	auto groupSize = exec.blockSize;
	auto blockSize = groupSize * exec.groupsPerBlock;
	unsigned int blockCount = (in.n + exec.groupsPerBlock - 1) / exec.groupsPerBlock;

	auto sharedMemory =
		sizeof(cg::block_tile_memory<1024>) + exec.groupsPerBlock * (in.dim + in.k * 2 + in.k + in.k * in.dim) * sizeof(F);

	if (sharedMemory > exec.sharedMemorySize)
		throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");

#define CASE_ProjectionBlockMultiKernel(THREADS)                                                                                                     \
	case THREADS:                                                                                                                                    \
		projectionBlockMultiKernel<F, BaseIndexer, THREADS><<<blockCount, blockSize, sharedMemory>>>(                                                \
			in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize, in.k, in.adjust, in.boost);                      \
		break

	switch (groupSize) {
		CASE_ProjectionBlockMultiKernel(4);
		CASE_ProjectionBlockMultiKernel(8);
		CASE_ProjectionBlockMultiKernel(16);
		CASE_ProjectionBlockMultiKernel(32);
		CASE_ProjectionBlockMultiKernel(64);
		CASE_ProjectionBlockMultiKernel(128);
		CASE_ProjectionBlockMultiKernel(256);
		CASE_ProjectionBlockMultiKernel(512);
		default:
			projectionBlockMultiKernel<F, BaseIndexer, 64><<<blockCount, 64, sharedMemory>>>(
				in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize, in.k, in.adjust, in.boost);
			break;
	}
}


template <typename F>
__inline__ __device__ F warpReduceSum(F val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xffffffff, val, offset);
	return val;
}


template <typename F, typename ArrayF>
__inline__ __device__ void readAligned(F* const __restrict__ dst, const F* const __restrict__ src, const TopkResult<F>* const __restrict__ neighbors,
									   const std::uint32_t n, const std::uint32_t dim, const std::uint32_t groupRank, const std::uint32_t groupSize)
{
	constexpr auto size = sizeof(ArrayF) / sizeof(F);

	const std::uint32_t loadsCount = dim / size;
	const std::uint32_t dimX = dim / size;

	ArrayF* const dstX = reinterpret_cast<ArrayF*>(dst);
	const ArrayF* const srcX = reinterpret_cast<const ArrayF*>(src);

	for (size_t i = groupRank; i < n * loadsCount; i += groupSize) {
		const auto idx = i / loadsCount;
		const auto off = i % loadsCount;

		dstX[idx * dimX + off] = srcX[neighbors[idx].index * dimX + off];
	}
}

template <typename F>
__inline__ __device__ void readAlignedGrid2D(F* const __restrict__ dst, const F* const __restrict__ src,
											 const TopkResult<F>* const __restrict__ neighbors, const std::uint32_t n, const std::uint32_t groupRank,
											 const std::uint32_t groupSize)
{
	using ArrayT = typename ArrayFloatType<F>::Type2;

	ArrayT* const dstX = reinterpret_cast<ArrayT*>(dst);
	const ArrayT* const srcX = reinterpret_cast<const ArrayT*>(src);

	for (size_t i = groupRank; i < n; i += groupSize)
		dstX[i] = srcX[neighbors[i].index];
}

template <typename F>
__inline__ __device__ void storeToCache(const std::uint32_t groupRank, const std::uint32_t groupSize, const F* const __restrict__ points,
										F* const __restrict__ pointsCache, const F* const __restrict__ grid, F* const __restrict__ gridCache,
										const F* const __restrict__ grid2d, F* const __restrict__ grid2dCache,
										const TopkResult<F>* const __restrict__ neighbors, const std::uint32_t dim,
										const std::uint32_t gridCacheLeadingDim, const std::uint32_t k)
{
	auto copyIdx = groupRank;
	for (; copyIdx < dim; copyIdx += groupSize)
		pointsCache[copyIdx] = points[copyIdx];

	readAlignedGrid2D<F>(grid2dCache, grid2d, neighbors, k, groupRank, groupSize);

	readAligned<F, typename ArrayFloatType<F>::Type4>(gridCache, grid, neighbors, k, gridCacheLeadingDim, groupRank, groupSize);
}

/**
 * One block computes embedding for multiple points.
 * Data in shared memory are aligned for better reading.
 */
template <typename F, typename INDEXER, size_t tileSize>
__global__ void projectionBlockMultiAlignedMemKernel(const F* __restrict__ points, const F* const __restrict__ grid,
													 const F* const __restrict__ grid2d, TopkResult<F>* __restrict__ neighbors,
													 F* __restrict__ projections, const std::uint32_t dim, const std::uint32_t n,
													 const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost,
													 const std::uint32_t groupSize, const std::uint32_t cacheLeadingDim)
{
	extern __shared__ char sharedMemory[];

	const std::uint32_t groupRank = threadIdx.x % groupSize;
	const std::uint32_t groupIdx = threadIdx.x / groupSize;
	const std::uint32_t groupsCount = blockDim.x / groupSize;

	const auto grid2dPadding = (k * 3) % cacheLeadingDim == 0 ? 0 : cacheLeadingDim - ((k * 3) % cacheLeadingDim);
	auto sharedMemoryoff = reinterpret_cast<F*>(sharedMemory) + ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * groupIdx;

	F* const __restrict__ pointCache = sharedMemoryoff;
	F* const __restrict__ gridCache = sharedMemoryoff + cacheLeadingDim;
	F* const __restrict__ grid2dCache = sharedMemoryoff + (k + 1) * cacheLeadingDim;
	F* const __restrict__ scoreCache = grid2dCache + k * 2;

	F* const __restrict__ reduceFinishStorage =
		reinterpret_cast<F*>(sharedMemory) + ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * groupsCount;

	auto tile = cg::tiled_partition<tileSize>(cg::this_thread_block());

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		const auto workIdx = blockIdx.x * groupsCount + groupIdx;

		if (workIdx >= n)
			return;

		points = points + workIdx * dim;
		neighbors = neighbors + workIdx * adjustedK;
		projections = projections + workIdx * 2;

		if (groupRank < tile.size())
			sortedDistsToScoresGroup<F>(tile, NeighborScoreStorage<F> { neighbors, scoreCache }, adjustedK, k, boost);
		else
			storeToCache(groupRank - tile.size(), groupSize - tile.size(), points, pointCache, grid, gridCache, grid2d, grid2dCache, neighbors, dim,
						 cacheLeadingDim, k);

		if (groupSize == tile.size())
			storeToCache(groupRank, groupSize, points, pointCache, grid, gridCache, grid2d, grid2dCache, neighbors, dim, cacheLeadingDim, k);


		__syncthreads();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = groupRank; i < k; i += groupSize)
		addGravity2Wise(scoreCache[i], grid2dCache + i * 2, mtx);

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = groupRank; i < neighborPairs; i += groupSize) {
		const auto indices = getIndices<INDEXER>(i, k);

		const auto I = indices.x;
		const auto J = indices.y;

		const auto result = euclideanProjection4Wise<F>(pointCache, gridCache + I * cacheLeadingDim, gridCache + J * cacheLeadingDim, dim);
		F scalarProjection = result.x;
		const F squaredGridPointsDistance = result.y;

		if (squaredGridPointsDistance == F(0))
			continue;

		scalarProjection /= squaredGridPointsDistance;

		addApproximation2Wise(scoreCache[I], scoreCache[J], grid2dCache + I * 2, grid2dCache + J * 2, adjust, scalarProjection, mtx);
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = cg::reduce(tile, mtx[i], cg::plus<F>());

		const auto warpId = threadIdx.x / warpSize;

		if (threadIdx.x % warpSize == 0 && groupRank != 0)
			reduceFinishStorage[warpId] = mtx[i];

		__syncthreads();

		if (groupRank == 0) {
			for (std::uint32_t j = 1; j < groupSize / warpSize; ++j) {
				mtx[i] += reduceFinishStorage[warpId + j];
			}
		}
	}

	if (groupRank == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionAlignedMemKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	auto groupSize = exec.blockSize;
	auto blockSize = groupSize * exec.groupsPerBlock;
	unsigned int blockCount = (in.n + exec.groupsPerBlock - 1) / exec.groupsPerBlock;
	auto warpCount = blockSize / 32;

	auto grid2dPadding = (in.k * 3) % in.gridCacheLeadingDim == 0 ? 0 : in.gridCacheLeadingDim - ((in.k * 3) % in.gridCacheLeadingDim);

	auto sharedMemory = sizeof(F) * warpCount + sizeof(F) * (in.k + 1) * in.gridCacheLeadingDim * exec.groupsPerBlock
						+ sizeof(F) * (in.k * 3 + grid2dPadding) * exec.groupsPerBlock;

	if (sharedMemory > exec.sharedMemorySize)
		throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");

#define CASE_ProjectionAlignedMemKernel(THREADS)                                                                                                     \
	case THREADS:                                                                                                                                    \
		projectionBlockMultiAlignedMemKernel<F, BaseIndexer, THREADS>                                                                                \
			<<<blockCount, blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize, in.k,  \
													  in.adjust, in.boost, groupSize, in.gridCacheLeadingDim);                                       \
		break

	switch (groupSize) {
		CASE_ProjectionAlignedMemKernel(2);
		CASE_ProjectionAlignedMemKernel(4);
		CASE_ProjectionAlignedMemKernel(8);
		CASE_ProjectionAlignedMemKernel(16);
		default:
			projectionBlockMultiAlignedMemKernel<F, RectangleIndexer, 32>
				<<<blockCount, blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize,
														  in.k, in.adjust, in.boost, groupSize, in.gridCacheLeadingDim);
			break;
	}
}

template <typename T>
__inline__ __device__ uint2 getPackedIndices(std::uint32_t plainIndex, std::uint32_t k, bool parity)
{
	const auto indices = getIndices<T>(plainIndex, k);

	if (parity)
		return { indices.x * 2, indices.y * 2 };
	else
		return { indices.x * 2, indices.y * 2 + 1 };
}

/**
 * One block computes embedding for multiple points.
 * Data in shared memory are aligned for better reading.
 * Thread caches 4 grid points to perform 4 projections while having them inside registers.
 */
template <typename F, typename INDEXER, size_t tileSize>
__global__ void projectionBlockMultiAlignedMemRegisterKernel(const F* __restrict__ points, const F* const __restrict__ grid,
															 const F* const __restrict__ grid2d, TopkResult<F>* __restrict__ neighbors,
															 F* __restrict__ projections, const std::uint32_t dim, const std::uint32_t n,
															 const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost,
															 const std::uint32_t groupSize, const std::uint32_t cacheLeadingDim)
{
	extern __shared__ char sharedMemory[];

	const std::uint32_t groupRank = threadIdx.x % groupSize;
	const std::uint32_t groupIdx = threadIdx.x / groupSize;
	const std::uint32_t groupsCount = blockDim.x / groupSize;

	const auto grid2dPadding = (k * 3) % cacheLeadingDim == 0 ? 0 : cacheLeadingDim - ((k * 3) % cacheLeadingDim);
	auto sharedMemoryoff = reinterpret_cast<F*>(sharedMemory) + ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * groupIdx;

	F* const __restrict__ pointCache = sharedMemoryoff;
	F* const __restrict__ gridCache = sharedMemoryoff + cacheLeadingDim;
	F* const __restrict__ grid2dCache = sharedMemoryoff + (k + 1) * cacheLeadingDim;
	F* const __restrict__ scoreCache = grid2dCache + k * 2;

	F* const __restrict__ reduceFinishStorage =
		reinterpret_cast<F*>(sharedMemory) + ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * groupsCount;

	auto tile = cg::tiled_partition<tileSize>(cg::this_thread_block());

	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		const auto workIdx = blockIdx.x * groupsCount + groupIdx;

		if (workIdx >= n)
			return;

		points = points + workIdx * dim;
		neighbors = neighbors + workIdx * adjustedK;
		projections = projections + workIdx * 2;

		if (groupRank < tile.size())
			sortedDistsToScoresGroup<F>(tile, NeighborScoreStorage<F> { neighbors, scoreCache }, adjustedK, k, boost);
		else
			storeToCache(groupRank - tile.size(), groupSize - tile.size(), points, pointCache, grid, gridCache, grid2d, grid2dCache, neighbors, dim,
						 cacheLeadingDim, k);

		if (groupSize == tile.size())
			storeToCache(groupRank, groupSize, points, pointCache, grid, gridCache, grid2d, grid2dCache, neighbors, dim, cacheLeadingDim, k);


		__syncthreads();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = groupRank; i < k; i += groupSize)
		addGravity2Wise(scoreCache[i], grid2dCache + i * 2, mtx);

	const auto packedK = k / 2;
	const auto packedNeighborPairs = (packedK * (packedK - 1)) / 2;

	for (std::uint32_t i = groupRank; i < packedNeighborPairs; i += groupSize) {
		const auto indices = getPackedIndices<INDEXER>(i, packedK, k % 2 == 0);

		const auto& I = indices.x;
		const auto& J = indices.y;

		typename ArrayFloatType<F>::Type2 packedResult[4] { typename ArrayFloatType<F>::Type2 { 0.0, 0.0 },
															typename ArrayFloatType<F>::Type2 { 0.0, 0.0 },
															typename ArrayFloatType<F>::Type2 { 0.0, 0.0 },
															typename ArrayFloatType<F>::Type2 { 0.0, 0.0 } };

		euclideanProjection4WiseRegister<F>(pointCache, gridCache + I * cacheLeadingDim, gridCache + (I + 1) * cacheLeadingDim,
											gridCache + J * cacheLeadingDim, gridCache + (J + 1) * cacheLeadingDim, dim, packedResult);

		#pragma unroll
		for (std::uint32_t packedI = 0; packedI < 4; packedI++) {
			F scalarProjection = packedResult[packedI].x;
			const F squaredGridPointsDistance = packedResult[packedI].y;

			if (squaredGridPointsDistance == F(0))
				continue;

			scalarProjection /= squaredGridPointsDistance;

			const auto unpackedI = I + packedI / 2;
			const auto unpackedJ = J + (packedI % 2);

			addApproximation2Wise(scoreCache[unpackedI], scoreCache[unpackedJ], grid2dCache + unpackedI * 2, grid2dCache + unpackedJ * 2, adjust, scalarProjection, mtx);
		}
	}

	if (k % 2 == 0) {
		for (std::uint32_t i = groupRank; i < packedK; i += groupSize) {
			const auto I = i * 2;
			const auto J = i * 2 + 1;

			const auto result = euclideanProjection4Wise<F>(pointCache, gridCache + I * cacheLeadingDim, gridCache + J * cacheLeadingDim, dim);
			F scalarProjection = result.x;
			const F squaredGridPointsDistance = result.y;

			if (squaredGridPointsDistance == F(0))
				continue;

			scalarProjection /= squaredGridPointsDistance;

			addApproximation2Wise(scoreCache[I], scoreCache[J], grid2dCache + I * 2, grid2dCache + J * 2, adjust, scalarProjection, mtx);
		}
	}
	else {
		for (std::uint32_t i = groupRank; i < packedK; i += groupSize) {
			const auto& I = i * 2;

			typename ArrayFloatType<F>::Type2 packedResult[3] { typename ArrayFloatType<F>::Type2 { 0.0, 0.0 },
																typename ArrayFloatType<F>::Type2 { 0.0, 0.0 },
																typename ArrayFloatType<F>::Type2 { 0.0, 0.0 }};

			euclideanProjection4WiseRegister<F>(pointCache, gridCache + I * cacheLeadingDim, gridCache + (I + 1) * cacheLeadingDim,
												gridCache + (I + 2) * cacheLeadingDim, dim, packedResult);

			#pragma unroll
			for (std::uint32_t packedI = 0; packedI < 3; packedI++) {
				F scalarProjection = packedResult[packedI].x;
				const F squaredGridPointsDistance = packedResult[packedI].y;

				if (squaredGridPointsDistance == F(0))
					continue;

				scalarProjection /= squaredGridPointsDistance;

				const auto unpackedI = I + packedI / 2;
				const auto unpackedJ = I + 1 + (packedI + 1) / 2;

				addApproximation2Wise(scoreCache[unpackedI], scoreCache[unpackedJ], grid2dCache + unpackedI * 2, grid2dCache + unpackedJ * 2, adjust,
									  scalarProjection, mtx);
			}
		}
	}


	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		const auto warpId = threadIdx.x / warpSize;
		mtx[i] = cg::reduce(tile, mtx[i], cg::plus<F>());

		if (threadIdx.x % warpSize == 0 && groupRank != 0)
			reduceFinishStorage[warpId] = mtx[i];

		__syncthreads();

		if (groupRank == 0) {
			for (std::uint32_t j = 1; j < groupSize / warpSize; ++j) {
				mtx[i] += reduceFinishStorage[warpId + j];
			}
		}
	}

	if (groupRank == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionAlignedMemRegisterKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	auto groupSize = exec.blockSize;
	auto blockSize = groupSize * exec.groupsPerBlock;
	unsigned int blockCount = (in.n + exec.groupsPerBlock - 1) / exec.groupsPerBlock;
	auto warpCount = blockSize / 32;

	auto grid2dPadding = (in.k * 3) % in.gridCacheLeadingDim == 0 ? 0 : in.gridCacheLeadingDim - ((in.k * 3) % in.gridCacheLeadingDim);

	auto sharedMemory = sizeof(F) * warpCount + sizeof(F) * (in.k + 1) * in.gridCacheLeadingDim * exec.groupsPerBlock
						+ sizeof(F) * (in.k * 3 + grid2dPadding) * exec.groupsPerBlock;

	if (sharedMemory > exec.sharedMemorySize)
		throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");

#define CASE_ProjectionAlignedMemRegisterKernel(THREADS)                                                                                             \
	case THREADS:                                                                                                                                    \
		projectionBlockMultiAlignedMemRegisterKernel<F, BaseIndexer, THREADS>                                                                        \
			<<<blockCount, blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize, in.k,  \
													  in.adjust, in.boost, groupSize, in.gridCacheLeadingDim);                                       \
		break

	switch (groupSize) {
		CASE_ProjectionAlignedMemRegisterKernel(2);
		CASE_ProjectionAlignedMemRegisterKernel(4);
		CASE_ProjectionAlignedMemRegisterKernel(8);
		CASE_ProjectionAlignedMemRegisterKernel(16);
		default:
			projectionBlockMultiAlignedMemRegisterKernel<F, BaseIndexer, 32>
				<<<blockCount, blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize,
														  in.k, in.adjust, in.boost, groupSize, in.gridCacheLeadingDim);
			break;
	}
}

/**
 * Modified projectionBlockMultiAlignedMem kernel.
 */
template <typename F, typename INDEXER, size_t tileSize>
__global__ void projectionBlockMultiAlignedMemSmallKernel(const F* __restrict__ points, const F* const __restrict__ grid,
														  const F* const __restrict__ grid2d, TopkResult<F>* __restrict__ neighbors,
														  F* __restrict__ projections, const std::uint32_t dim, const std::uint32_t n,
														  const std::uint32_t gridSize, const std::uint32_t k, const F adjust, const F boost,
														  const std::uint32_t groupSize, const std::uint32_t cacheLeadingDim)
{
	extern __shared__ char sharedMem[];
	auto* __restrict__ block_mem = reinterpret_cast<cg::block_tile_memory<1024>*>(sharedMem);
	F* __restrict__ sharedMemStart = reinterpret_cast<F*>(block_mem + 1);

	auto block = cg::this_thread_block(*block_mem);
	auto group = cg::tiled_partition<tileSize>(block);

	const auto grid2dPadding = (k * 3) % cacheLeadingDim == 0 ? 0 : cacheLeadingDim - ((k * 3) % cacheLeadingDim);
	auto sharedMemoryoff = reinterpret_cast<F*>(sharedMemStart) + ((k + 1) * cacheLeadingDim + k * 3 + grid2dPadding) * group.meta_group_rank();

	F* const __restrict__ pointCache = sharedMemoryoff;
	F* const __restrict__ gridCache = sharedMemoryoff + cacheLeadingDim;
	F* const __restrict__ grid2dCache = sharedMemoryoff + (k + 1) * cacheLeadingDim;
	F* const __restrict__ scoreCache = grid2dCache + k * 2;


	// assign defaults and generate scores
	{
		const std::uint32_t adjustedK = k < gridSize ? k + 1 : k;

		const auto workIdx = blockIdx.x * group.meta_group_size() + group.meta_group_rank();

		if (workIdx >= n)
			return;

		points = points + workIdx * dim;
		neighbors = neighbors + workIdx * adjustedK;
		projections = projections + workIdx * 2;

		sortedDistsToScoresGroup<F>(group, NeighborScoreStorage<F> { neighbors, scoreCache }, adjustedK, k, boost);

		storeToCache(group.thread_rank(), group.size(), points, pointCache, grid, gridCache, grid2d, grid2dCache, neighbors, dim, cacheLeadingDim, k);

		group.sync();
	}

	F mtx[6];
	memset(mtx, 0, 6 * sizeof(F));

	for (std::uint32_t i = group.thread_rank(); i < k; i += group.size())
		addGravity2Wise(scoreCache[i], grid2dCache + i * 2, mtx);

	const std::uint32_t neighborPairs = (k * (k - 1)) / 2;
	for (std::uint32_t i = group.thread_rank(); i < neighborPairs; i += group.size()) {
		const auto indices = getIndices<INDEXER>(i, k);

		const auto I = indices.x;
		const auto J = indices.y;

		const auto result = euclideanProjection4Wise<F>(pointCache, gridCache + I * cacheLeadingDim, gridCache + J * cacheLeadingDim, dim);
		F scalarProjection = result.x;
		const F squaredGridPointsDistance = result.y;

		if (squaredGridPointsDistance == F(0))
			continue;

		scalarProjection /= squaredGridPointsDistance;

		addApproximation2Wise(scoreCache[I], scoreCache[J], grid2dCache + I * 2, grid2dCache + J * 2, adjust, scalarProjection, mtx);
	}

	#pragma unroll
	for (size_t i = 0; i < 6; ++i) {
		mtx[i] = cg::reduce(group, mtx[i], cg::plus<F>());
	}

	if (group.thread_rank() == 0) {
		const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
		projections[0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
		projections[1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
	}
}

// runner wrapped in a class
template <typename F>
void ProjectionAlignedMemSmallKernel<F>::run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec)
{
	auto groupSize = exec.blockSize;
	auto blockSize = groupSize * exec.groupsPerBlock;
	unsigned int blockCount = (in.n + exec.groupsPerBlock - 1) / exec.groupsPerBlock;
	auto warpCount = blockSize / 32;

	auto grid2dPadding = (in.k * 3) % in.gridCacheLeadingDim == 0 ? 0 : in.gridCacheLeadingDim - ((in.k * 3) % in.gridCacheLeadingDim);

	auto sharedMemory = sizeof(cg::block_tile_memory<1024>) + sizeof(F) * warpCount
						+ sizeof(F) * (in.k + 1) * in.gridCacheLeadingDim * exec.groupsPerBlock
						+ sizeof(F) * (in.k * 3 + grid2dPadding) * exec.groupsPerBlock;

	if (sharedMemory > exec.sharedMemorySize)
		throw std::runtime_error("Insufficient size of shared memory for selected CUDA parameters.");

#define CASE_ProjectionBlockMultiSmallKernel(THREADS)                                                                                                \
	case THREADS:                                                                                                                                    \
		projectionBlockMultiAlignedMemSmallKernel<F, BaseIndexer, THREADS>                                                                           \
			<<<blockCount, blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize, in.k,  \
													  in.adjust, in.boost, groupSize, in.gridCacheLeadingDim);                                       \
		break

	switch (groupSize) {
		CASE_ProjectionBlockMultiSmallKernel(8);
		CASE_ProjectionBlockMultiSmallKernel(16);
		CASE_ProjectionBlockMultiSmallKernel(32);
		CASE_ProjectionBlockMultiSmallKernel(64);
		CASE_ProjectionBlockMultiSmallKernel(128);
		CASE_ProjectionBlockMultiSmallKernel(256);
		CASE_ProjectionBlockMultiSmallKernel(512);
		default:
			projectionBlockMultiAlignedMemSmallKernel<F, BaseIndexer, 64>
				<<<blockCount, blockSize, sharedMemory>>>(in.points, in.grid, in.grid2d, in.neighbors, in.projections, in.dim, in.n, in.gridSize,
														  in.k, in.adjust, in.boost, groupSize, in.gridCacheLeadingDim);
			break;
	}
}


/*
 * Explicit template instantiation.
 */
template <typename F>
void instantiateKernelRunnerTemplates()
{
	ProjectionProblemInstance<F> instance(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, F(0), F(0));
	CudaExecParameters exec;

	ProjectionBaseKernel<F>::run(instance, exec);
	ProjectionBlockKernel<F>::run(instance, exec);
	ProjectionBlockRectangleIndexKernel<F>::run(instance, exec);
	ProjectionBlockSharedKernel<F>::run(instance, exec);
	ProjectionBlockMultiKernel<F>::run(instance, exec);
	ProjectionAlignedMemKernel<F>::run(instance, exec);
	ProjectionAlignedMemRegisterKernel<F>::run(instance, exec);
	ProjectionAlignedMemSmallKernel<F>::run(instance, exec);
}

template void instantiateKernelRunnerTemplates<float>();
#ifndef NO_DOUBLES
template void instantiateKernelRunnerTemplates<double>();
#endif
