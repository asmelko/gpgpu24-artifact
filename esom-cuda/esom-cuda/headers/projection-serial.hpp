#ifndef ESOM_CUDA_PROJECTION_SERIAL_HPP
#define ESOM_CUDA_PROJECTION_SERIAL_HPP

#include "interface.hpp"


template <typename F = float>
class ProjectionSerialAlgorithm : public IProjectionAlgorithm<F>
{
public:
	using TopkResult = typename IProjectionAlgorithm<F>::TopkResult;

private:
	const DataPoints<F> *mPoints, *mGrid, *mGrid2D;
	const std::vector<TopkResult>* mNeighborsPtr;
	std::vector<TopkResult> mNeighbors;

public:
	void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2D, const std::vector<TopkResult>& neighbors,
					const std::size_t k, const F adjust, const F boost) override
	{
		IProjectionAlgorithm<F>::initialize(points, grid, grid2D, neighbors, k, adjust, boost);
		mPoints = &points;
		mGrid = &grid;
		mGrid2D = &grid2D;
		mNeighborsPtr = &neighbors;
	}

	void prepareInputs() override 
	{
		IProjectionAlgorithm<F>::prepareInputs();
		mNeighbors = *mNeighborsPtr;
	}

	void run() override
	{
		const std::size_t adjustedK = this->mK < this->mGridSize ? this->mK + 1 : this->mK;

		for (std::size_t pointIdx = 0; pointIdx < this->mN; ++pointIdx) 
		{
			const F* point = (*this->mPoints)[pointIdx];
			auto neighbors = this->mNeighbors.data() + pointIdx * adjustedK;

			sortedDistsToScores(neighbors, adjustedK);

			// create the empty equation matrix of (2 * (1 + 2)) zeros
			F mtx[6];
			std::fill(mtx, mtx + 6, F(0));

			// for all points in the neighborhood
			for (std::size_t i = 0; i < this->mK; ++i) 
			{
				const std::size_t idxI = neighbors[i].index;
				const F scoreI = neighbors[i].distance;

				/* this adds a really tiny influence of the point to
				 * prevent singularities */
				addGravity((*this->mGrid2D)[idxI], scoreI, mtx);

				// for all combinations of point 'i' with points in the
				// neighborhood
				for (std::size_t j = i + 1; j < this->mK; ++j) {
					const std::size_t idxJ = neighbors[j].index;
					const F scoreJ = neighbors[j].distance;

					F scalarProjection, squaredGridPointsDistance;
					euclideanProjection((*this->mGrid)[idxI], (*this->mGrid)[idxJ], point, this->mDim, scalarProjection, squaredGridPointsDistance);

					if (squaredGridPointsDistance == F(0))
						continue;

					scalarProjection /= squaredGridPointsDistance;

					addApproximation(scoreI, scoreJ, (*this->mGrid2D)[idxI], (*this->mGrid2D)[idxJ], scalarProjection, mtx);
				}
			}

			// solve linear equation
			const F det = mtx[0] * mtx[3] - mtx[1] * mtx[2];
			this->mResult[pointIdx][0] = (mtx[4] * mtx[3] - mtx[5] * mtx[2]) / det;
			this->mResult[pointIdx][1] = (mtx[0] * mtx[5] - mtx[1] * mtx[4]) / det;
		}
	}

private:

	static F euclideanSquareDistance(const F* lhs, const F* rhs, const std::size_t dim)
	{
		F squareDistance = 0;
		for (std::size_t i = 0; i < dim; ++i) {
			F tmp = lhs[i] - rhs[i];
			squareDistance += tmp * tmp;
		}
		return squareDistance;
	}

	static void euclideanProjection(const F* gridPoint1, const F* gridPoint2, const F* point, const std::size_t dim, F& oScalarProjection,
									F& oSquareGridPointDistance)
	{
		F scalarProjection = 0;
		F squareGridPointDistance = 0;

		for (std::size_t i = 0; i < dim; ++i) {
			auto tmp = gridPoint2[i] - gridPoint1[i];
			squareGridPointDistance += tmp * tmp;
			scalarProjection += tmp * (point[i] - gridPoint1[i]);
		}

		oScalarProjection = scalarProjection;
		oSquareGridPointDistance = squareGridPointDistance;
	}

	void sortedDistsToScores(TopkResult* neighbors, const size_t adjustedK)
	{
		// compute the distance distribution for the scores
		F mean = 0, sd = 0, wsum = 0;
		for (size_t i = 0; i < adjustedK; ++i) {
			wsum += 1 / F(i + 1);
		}

		for (size_t i = 0; i < adjustedK; ++i) {
			neighbors[i].distance = std::sqrt(neighbors[i].distance);
		}

		for (size_t i = 0; i < adjustedK; ++i) {
			const F tmp = neighbors[i].distance;
			const F w = 1 / F(i + 1);
			mean += tmp * w;
		}

		mean /= wsum;

		for (size_t i = 0; i < adjustedK; ++i) {
			const F tmp = neighbors[i].distance - mean;
			const F w = 1 / F(i + 1);
			sd += tmp * tmp * w;
		}

		sd = this->mBoost / std::sqrt(sd / wsum);
		const F nmax = ProjectionProblemInstance<F>::maxAvoidance / neighbors[adjustedK - 1].distance;

		// convert the stuff to scores
		for (size_t i = 0; i < this->mK; ++i) {
			if (this->mK < adjustedK)
				neighbors[i].distance = std::exp((mean - neighbors[i].distance) * sd)
										* (1 - std::exp(neighbors[i].distance * nmax - ProjectionProblemInstance<F>::maxAvoidance));
			else
				neighbors[i].distance = std::exp((mean - neighbors[i].distance) * sd);
		}
	}

	void addGravity(const F* grid2DPoint, const F score, F* mtx)
	{
		const F gs = score * ProjectionProblemInstance<F>::gridGravity;

		mtx[0] += gs;
		mtx[3] += gs;
		mtx[4] += gs * grid2DPoint[0];
		mtx[5] += gs * grid2DPoint[1];
	}

	void addApproximation(const F scoreI, const F scoreJ, const F* grid2DPointI, const F* grid2DPointJ, F scalarProjection, F* mtx)
	{
		F h[2], hp = 0;
		for (std::size_t i = 0; i < 2; ++i) {
			// hp += sqr(h[i] = jec[i] - iec[i]);
			h[i] = grid2DPointJ[i] - grid2DPointI[i];
			hp += h[i] * h[i];
		}

		if (hp < ProjectionProblemInstance<F>::zeroAvoidance)
			return;

		const F s = scoreI * scoreJ * std::pow<F>(1 + hp, -this->mAdjust) * (F)std::exp(-(scalarProjection - .5) * (scalarProjection - .5));
		const F sihp = s / hp;
		const F rhsc = s * (scalarProjection + (h[0] * grid2DPointI[0] + h[1] * grid2DPointI[1]) / hp);

		mtx[0] += h[0] * h[0] * sihp;
		mtx[1] += h[0] * h[1] * sihp;
		mtx[2] += h[1] * h[0] * sihp;
		mtx[3] += h[1] * h[1] * sihp;
		mtx[4] += h[0] * rhsc;
		mtx[5] += h[1] * rhsc;
	}
};


#endif
