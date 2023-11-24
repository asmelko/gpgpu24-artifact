#ifndef ESOM_CUDA_TOPK_SERIAL_HPP
#define ESOM_CUDA_TOPK_SERIAL_HPP

#include "interface.hpp"

#include <algorithm>

/**
 * Serial implementation of top-k algorithm.
 */
template<typename F = float>
class TopkSerialAlgorithm : public ITopkAlgorithm<F>
{
public:
	using Result = typename ITopkAlgorithm<F>::Result;

private:
	const DataPoints<F> *mPoints, *mGrid;
	
	void insertSortStep(Result *results, std::size_t idx)
	{
		while (idx > 0 && results[idx].distance < results[idx-1].distance) {
			std::swap(results[idx], results[idx - 1]);
			--idx;
		}
	}

	F distance(std::size_t point, std::size_t gridPoint) const
	{
		F sum = (F)0.0;
		const F* p = (*this->mPoints)[point];
		const F* g = (*this->mGrid)[gridPoint];
		for (std::size_t d = 0; d < this->mDim; ++d) {
			F diff = *p++ - *g++; // ah, smells like pure ol' C
			sum += diff * diff;
		}
		return sum; // squared euclidean
	}

public:
	TopkSerialAlgorithm() : mPoints(nullptr), mGrid(nullptr) {}

	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const std::size_t neighbors) override
	{
		ITopkAlgorithm<F>::initialize(points, grid, neighbors);
		mPoints = &points;
		mGrid = &grid;
	}

	void run() override
	{
		Result* currentTopK = this->mResult.data();
		for (std::size_t point = 0; point < this->mN; ++point) {
			for (std::size_t i = 0; i < this->mK; ++i) {
				currentTopK[i].distance = distance(point, i);
				currentTopK[i].index = (std::uint32_t)i;
				insertSortStep(currentTopK, i);
			}

			for (std::size_t i = this->mK; i < this->mGridSize; ++i) {
				F dist = distance(point, i);
				if (currentTopK[this->mK-1].distance > dist) {
					currentTopK[this->mK-1].distance = dist;
					currentTopK[this->mK-1].index = (std::uint32_t)i; 
					insertSortStep(currentTopK, this->mK-1);
				}
			}

			currentTopK += this->mK;
		}
	}
};


#endif
