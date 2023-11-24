#ifndef ESOM_CUDA_ESOM_SERIAL_HPP
#define ESOM_CUDA_ESOM_SERIAL_HPP

#include "interface.hpp"
#include "projection-serial.hpp"
#include "topk-serial.hpp"

template <typename F = float>
class EsomSerialAlgorithm : public IEsomAlgorithm<F>
{
	TopkSerialAlgorithm<F> mTopkAlgorithm;
	ProjectionSerialAlgorithm<F> mProjectionAlgorithm;

public:
	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2D,
					const std::size_t k, const F adjust, const F boost) override
	{
		mTopkAlgorithm.initialize(points, grid, k < grid.size() ? k + 1 : k);
		mProjectionAlgorithm.initialize(points, grid, grid2D, mTopkAlgorithm.getResults(), k, adjust, boost);
	}

	void prepareInputs() override
	{
		mTopkAlgorithm.prepareInputs();
	}

	void run() override
	{
		mTopkAlgorithm.run();
		mProjectionAlgorithm.prepareInputs();
		mProjectionAlgorithm.run();
	}

	const DataPoints<F>& getResults() override
	{
		return mProjectionAlgorithm.getResults();
	}

	void cleanup() override
	{
		mTopkAlgorithm.cleanup();
		mProjectionAlgorithm.cleanup();
	}
};


#endif
