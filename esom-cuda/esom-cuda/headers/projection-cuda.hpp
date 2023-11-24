#ifndef ESOM_CUDA_PROJECTION_CUDA_HPP
#define ESOM_CUDA_PROJECTION_CUDA_HPP


#include <algorithm>
#include <type_traits>

#include "interface.hpp"
#include "projection.cuh"
#include "structs.cuh"
#include <cuda/cuda.hpp>


/**
 * Basic CUDA implementation of top-k algorithm with templated kernel wrapper.
 */
template <typename F, class KERNEL>
class ProjectionCudaAlgorithm : public IProjectionAlgorithm<F>
{
public:
	using TopkResult = typename IProjectionAlgorithm<F>::TopkResult;

protected:
	bpp::CudaBuffer<F> mCuPoints;
	bpp::CudaBuffer<F> mCuGrid;
	bpp::CudaBuffer<F> mCuGrid2d;
	bpp::CudaBuffer<TopkResult> mCuNeighbors;
	bpp::CudaBuffer<F> mCuResult;

	const DataPoints<F>*mPoints, *mGrid, *mGrid2d;
	const std::vector<TopkResult>* mNeighbors;

	CudaExecParameters& mCudaExec;
	bool mResultLoaded;

	std::size_t getNeighborK() const
	{
		return this->mK < this->mGridSize ? this->mK + 1 : this->mK;
	}

	void resizeBuffers()
	{
		mCuPoints.realloc(this->mN * this->mDim);
		mCuGrid.realloc(this->mGridSize * this->mDim);
		mCuGrid2d.realloc(this->mGridSize * 2);
		mCuNeighbors.realloc(this->mN * getNeighborK());
		mCuResult.realloc(this->mN * 2);
	}

public:
	ProjectionCudaAlgorithm(CudaExecParameters& cudaExec)
		: mPoints(nullptr), mGrid(nullptr), mGrid2d(nullptr), mNeighbors(nullptr), mCudaExec(cudaExec), mResultLoaded(false)
	{}

	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2d, const std::vector<TopkResult>& neighbors,
					const std::size_t k, const F adjust, const F boost) override
	{
		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0) {
			throw bpp::RuntimeError("No CUDA devices found!");
		}

		IProjectionAlgorithm<F>::initialize(points, grid, grid2d, neighbors, k, adjust, boost);
		mPoints = &points;
		mGrid = &grid;
		mGrid2d = &grid2d;
		mNeighbors = &neighbors;
	}

	void prepareInputs() override
	{
		IProjectionAlgorithm<F>::prepareInputs();

		CUCH(cudaSetDevice(0));
		resizeBuffers();

		mCuPoints.write(this->mPoints->data(), this->mN * this->mDim);
		mCuGrid.write(this->mGrid->data(), this->mGridSize * this->mDim);
		mCuGrid2d.write(this->mGrid2d->data(), this->mGridSize * 2);
		mCuNeighbors.write(*this->mNeighbors, this->mN * getNeighborK());
		mResultLoaded = false;
	}


	void run() override
	{
		ProjectionProblemInstance<F> projectionProblem(*mCuPoints, *mCuGrid, *mCuGrid2d, *mCuNeighbors, *mCuResult, this->mDim, this->mN,
													   this->mGridSize, this->mK, this->mAdjust, this->mBoost);
		KERNEL::run(projectionProblem, mCudaExec);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
	}

	const DataPoints<F>& getResults() override
	{
		if (!mResultLoaded) {
			mCuResult.read(this->mResult.data());
			mResultLoaded = true;
		}

		return this->mResult;
	}

	void cleanup() override
	{
		IProjectionAlgorithm<F>::cleanup();
		mCuPoints.free();
		mCuGrid.free();
		mCuGrid2d.free();
		mCuNeighbors.free();
		mCuResult.free();
	}
};

template <typename F>
constexpr size_t getRequiredGrid2DAlignment() 
{
	if constexpr (std::is_same_v<F, float>)
		return 8;
	return 16;
}

/**
 * Basic CUDA implementation of top-k algorithm with templated kernel wrapper.
 */
template <typename F, class KERNEL>
class AlignedProjectionCudaAlgorithm : public ProjectionCudaAlgorithm<F, KERNEL>
{
	using TopkResult = typename IProjectionAlgorithm<F>::TopkResult;

	std::size_t mGridLeadingDim;

public:
	AlignedProjectionCudaAlgorithm(CudaExecParameters& cudaExec) : ProjectionCudaAlgorithm<F, KERNEL>(cudaExec), mGridLeadingDim(0)
	{}

	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2d, const std::vector<TopkResult>& neighbors,
					const std::size_t k, const F adjust, const F boost) override
	{
		ProjectionCudaAlgorithm<F, KERNEL>::initialize(points, grid, grid2d, neighbors, k, adjust, boost);

		constexpr size_t alignment = 16;

		auto pointBytes = this->mDim * sizeof(F);
		auto dimCacheResidual = pointBytes % alignment;
		mGridLeadingDim = pointBytes + (dimCacheResidual == 0 ? 0 : alignment - dimCacheResidual);
		mGridLeadingDim /= sizeof(F);
	}

	void prepareInputs() override
	{
		IProjectionAlgorithm<F>::prepareInputs();

		CUCH(cudaSetDevice(0));

		this->mCuPoints.realloc(this->mN * this->mDim);
		this->mCuGrid.realloc(this->mGridSize * mGridLeadingDim);
		this->mCuGrid2d.realloc(this->mGridSize * 2);
		this->mCuNeighbors.realloc(this->mN * this->getNeighborK());
		this->mCuResult.realloc(this->mN * 2);

		this->mCuPoints.write(this->mPoints->data(), this->mN * this->mDim);
		this->mCuNeighbors.write(*this->mNeighbors, this->mN * this->getNeighborK());
		this->mCuGrid2d.write(this->mGrid2d->data(), this->mGridSize * 2);

		CUCH(cudaMemcpy2D(*this->mCuGrid, mGridLeadingDim * sizeof(F), this->mGrid->data(), this->mDim * sizeof(F), this->mDim * sizeof(F),
						  this->mGridSize,
						  cudaMemcpyKind::cudaMemcpyHostToDevice));

		this->mResultLoaded = false;
	}


	void run() override
	{
		ProjectionProblemInstance<F> projectionProblem(*this->mCuPoints, *this->mCuGrid, *this->mCuGrid2d, *this->mCuNeighbors, *this->mCuResult,
													   this->mDim, this->mN, this->mGridSize, this->mK, this->mAdjust, this->mBoost, mGridLeadingDim);
		KERNEL::run(projectionProblem, this->mCudaExec);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
	}
};


#endif
