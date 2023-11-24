#ifndef ESOM_CUDA_TOPK_CUDA_HPP
#define ESOM_CUDA_TOPK_CUDA_HPP


#include <algorithm>

#include "interface.hpp"
#include "structs.cuh"
#include "topk.cuh"
#include <cuda/cuda.hpp>


/**
 * Basic CUDA implementation of top-k algorithm with templated kernel wrapper.
 */
template <typename F, class KERNEL>
class TopkCudaAlgorithm : public ITopkAlgorithm<F>
{
public:
	using Result = typename ITopkAlgorithm<F>::Result;

protected:
	bpp::CudaBuffer<F> mCuPoints;
	bpp::CudaBuffer<F> mCuGrid;
	bpp::CudaBuffer<Result> mCuResult;

private:
	CudaExecParameters& mCudaExec;
	const DataPoints<F>*mPoints, *mGrid;
	bool mResultLoaded;

	void resizeBuffers()
	{
		mCuPoints.realloc(this->mN * this->mDim);
		mCuGrid.realloc(this->mGridSize * this->mDim);
		mCuResult.realloc(this->mN * this->mK);
	}

public:
	TopkCudaAlgorithm(CudaExecParameters& cudaExec) : mCudaExec(cudaExec), mPoints(nullptr), mGrid(nullptr), mResultLoaded(false) {}

	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const std::size_t neighbors) override
	{
		std::size_t devices = bpp::CudaDevice::count();
		if (devices == 0) {
			throw bpp::RuntimeError("No CUDA devices found!");
		}

		ITopkAlgorithm<F>::initialize(points, grid, neighbors);
		mPoints = &points;
		mGrid = &grid;
	}

	void prepareInputs() override
	{
		ITopkAlgorithm<F>::prepareInputs();
		CUCH(cudaSetDevice(0));
		resizeBuffers();

		mCuPoints.write(this->mPoints->data(), this->mN * this->mDim);
		mCuGrid.write(this->mGrid->data(), this->mGridSize * this->mDim);
		mResultLoaded = false;
	}


	void run() override
	{
		TopkProblemInstance<F> topkProblem(*mCuPoints, *mCuGrid, *mCuResult, this->mDim, this->mN, this->mGridSize, this->mK);
		KERNEL::run(topkProblem, mCudaExec);
		CUCH(cudaGetLastError());
		CUCH(cudaDeviceSynchronize());
	}

	const std::vector<Result>& getResults() override
	{
		if (!mResultLoaded) {
			mCuResult.read(this->mResult);
			mResultLoaded = true;
		}

		return this->mResult;
	}

	void cleanup() override
	{
		ITopkAlgorithm<F>::cleanup();
		mCuPoints.free();
		mCuGrid.free();
		mCuResult.free();
	}
};


#endif
