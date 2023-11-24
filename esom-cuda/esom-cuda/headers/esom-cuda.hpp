#ifndef ESOM_CUDA_ESOM_CUDA_HPP
#define ESOM_CUDA_ESOM_CUDA_HPP

#include "interface.hpp"
#include "projection-cuda.hpp"
#include "topk-cuda.hpp"

template <typename F, class TOPK_KERNEL, class PROJECTION_KERNEL>
class EsomCudaAlgorithm : public IEsomAlgorithm<F>
{
	class TopkAlgorithm : public TopkCudaAlgorithm<F, TOPK_KERNEL>
	{
	public:
		TopkAlgorithm(CudaExecParameters& cudaExec) : TopkCudaAlgorithm<F, TOPK_KERNEL>(cudaExec) {}

		friend EsomCudaAlgorithm<F, TOPK_KERNEL, PROJECTION_KERNEL>;
	};


	class ProjectionAlgorithm : public ProjectionCudaAlgorithm<F, PROJECTION_KERNEL>
	{
		using TopkResult = typename IProjectionAlgorithm<F>::TopkResult;

		const DataPoints<F>* mGrid2d;

	public:
		ProjectionAlgorithm(CudaExecParameters& cudaExec) : ProjectionCudaAlgorithm<F, PROJECTION_KERNEL>(cudaExec), mGrid2d(nullptr) {}

		void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2d, const std::size_t k, const F adjust,
						const F boost)
		{
			std::size_t devices = bpp::CudaDevice::count();
			if (devices == 0) {
				throw bpp::RuntimeError("No CUDA devices found!");
			}

			static std::vector<typename TopkProblemInstance<F>::Result> neighbors;
			IProjectionAlgorithm<F>::initialize(points, grid, grid2d, neighbors, k, adjust, boost);

			mGrid2d = &grid2d;
		}

		void moveBuffers(bpp::CudaBuffer<F>& points, bpp::CudaBuffer<F>& grid, bpp::CudaBuffer<typename TopkProblemInstance<F>::Result>& neighbors)
		{
			this->mCuPoints = std::move(points);
			this->mCuGrid = std::move(grid);
			this->mCuNeighbors = std::move(neighbors);
			points = bpp::CudaBuffer<F>();
			grid = bpp::CudaBuffer<F>();
			neighbors = bpp::CudaBuffer<typename TopkProblemInstance<F>::Result>();
		}

		void prepareInputs() override
		{
			IProjectionAlgorithm<F>::prepareInputs();
			this->mCuGrid2d.realloc(this->mGridSize * 2);
			this->mCuResult.realloc(this->mN * 2);

			this->mCuGrid2d.write(mGrid2d->data(), this->mGridSize * 2);
		}
	};

	TopkAlgorithm mTopkAlgorithm;
	ProjectionAlgorithm mProjectionAlgorithm;

public:
	EsomCudaAlgorithm(CudaExecParameters& cudaExec) : mTopkAlgorithm(cudaExec), mProjectionAlgorithm(cudaExec) {}

	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2D, const std::size_t k, const F adjust,
					const F boost) override
	{
		mTopkAlgorithm.initialize(points, grid, k < grid.size() ? k + 1 : k);
		mProjectionAlgorithm.initialize(points, grid, grid2D, k, adjust, boost);
	}

	void run() override
	{
		mTopkAlgorithm.run();
		mProjectionAlgorithm.moveBuffers(mTopkAlgorithm.mCuPoints, mTopkAlgorithm.mCuGrid, mTopkAlgorithm.mCuResult);
		mProjectionAlgorithm.run();
	}

	void prepareInputs() override
	{
		mTopkAlgorithm.prepareInputs();
		mProjectionAlgorithm.prepareInputs();
	}

	void cleanup() override
	{
		mTopkAlgorithm.cleanup();
		mProjectionAlgorithm.cleanup();
	}

	const DataPoints<F>& getResults() override
	{
		return mProjectionAlgorithm.getResults();
	}
};


#endif
