#ifndef ESOM_CUDA_INTERFACE_HPP
#define ESOM_CUDA_INTERFACE_HPP

#include <cstdint>
#include <iostream>
#include <vector>

#include <misc/exception.hpp>

#include "points.hpp"
#include "structs.cuh"


/**
 * Interface (base class) for all ESOM algorithms.
 */
template <typename F = float>
class IEsomAlgorithm
{
protected:
	std::size_t mDim, mN, mGridSize, mK;
	F mAdjust, mBoost;

public:
	IEsomAlgorithm() : mDim(0), mN(0), mGridSize(0), mK(0), mAdjust(0), mBoost(0) {}

	virtual ~IEsomAlgorithm() {}

	virtual void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2D,
							const std::size_t k, const F adjust, const F boost)
	{
		mDim = points.getDim();
		mN = points.size();

		if (grid.getDim() != mDim) {
			throw(bpp::RuntimeError() << "Data points and grid points have different dimensions: " << mDim << " != " << grid.getDim());
		}
		mGridSize = grid.size();

		if (grid2D.getDim() != 2) {
			throw(bpp::RuntimeError() << "Only 2D projection of the grid is currently supported (" << grid2D.getDim() << " dimensions given).");
		}
		if (grid2D.size() != mGridSize) {
			throw(bpp::RuntimeError() << "High-dimensional and low-dimensional grid coordinates have different sizes: " << mGridSize
									  << " != " << grid2D.size());
		}

		mK = k;
		mAdjust = adjust;
		mBoost = boost;

		// Derived class should save the pointers to inputs or copy them...
		// This part of the algorithm is not measured
	}

	virtual void prepareInputs()
	{
		// Transpose input data, copy them to GPU, ...
		// This part is measured separately from the run.
	}

	virtual void run() = 0;

	// Fetch the results (possibly from GPU memory).
	virtual const DataPoints<F>& getResults() = 0;

	bool verifyResult(IEsomAlgorithm<F>& refAlgorithm, std::ostream& out)
	{
		refAlgorithm.prepareInputs();
		refAlgorithm.run();
		auto refResult = refAlgorithm.getResults();
		auto result = getResults();

		std::size_t errors = 0;
		for (std::size_t p = 0; p < this->mN; ++p) {
			std::size_t i = 0;
			while (i < 2 && compare_floats_relative(refResult[p][i], result[p][i]) < 0.001)
				++i;

			if (i < 2) {
				if (++errors < 16) { // only first 16 errors are printed out
					out << "Error [" << p << "]:" << std::endl;
					for (std::size_t j = 0; j < 2; ++j)
						out << "\t[" << j << "]";
					out << std::endl;

					for (std::size_t j = 0; j < 2; ++j)
						out << "\t" << result[p][j];
					out << std::endl;

					for (std::size_t j = 0; j < 2; ++j)
						out << "\t" << refResult[p][j];
					out << std::endl;
				}
			}
		}

		if (errors) {
			out << "Total errors: " << errors << std::endl;
		}
		else {
			out << "Verification OK." << std::endl;
		}

		refAlgorithm.cleanup();
		return errors == 0;
	}

	virtual void cleanup() = 0;
};



/**
 * Interface (base class) for all top-k algorithms (first phase of ESOM).
 */
template <typename F = float>
class ITopkAlgorithm
{
public:
	using Result = typename TopkProblemInstance<F>::Result;

protected:
	std::size_t mDim, mN, mGridSize, mK;
	std::vector<Result> mResult; // n x k

public:
	ITopkAlgorithm() : mDim(0), mN(0), mGridSize(0), mK(0), mResult(0) {}

	virtual ~ITopkAlgorithm() {}

	virtual void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const std::size_t neighbors)
	{
		mDim = points.getDim();
		mN = points.size();

		if (grid.getDim() != mDim) {
			throw(bpp::RuntimeError() << "Data points and grid points have different dimensions: " << mDim << " != " << grid.getDim());
		}
		mGridSize = grid.size();

		mK = neighbors;

		// Derived class should save the pointers to inputs or copy them...
		// This part of the algorithm is not measured
	}

	virtual void prepareInputs()
	{
		mResult.resize(mN * mK);
		// Transpose input data, copy them to GPU, ...
		// This part is measured separately from the run.
	}

	virtual void run() = 0;

	virtual const std::vector<Result>& getResults()
	{
		// Fetch the results (possibly from GPU memory).

		return mResult;
	}

	bool verifyResult(ITopkAlgorithm<F>& refAlgorithm, std::ostream& out)
	{
		refAlgorithm.prepareInputs();
		refAlgorithm.run();
		auto refResult = refAlgorithm.getResults();
		auto result = getResults();

		std::size_t errors = 0;
		for (std::size_t p = 0; p < this->mN; ++p) {
			std::size_t offset = p * this->mK;

			std::size_t i = 0;
			while (i < this->mK && refResult[offset + i].index == result[offset + i].index
				   && compare_floats_relative(refResult[offset + i].distance, result[offset + i].distance) < 0.001)
				++i;

			if (i < this->mK) {
				if (++errors < 16) { // only first 16 errors are printed out
					out << "Error [" << p << "]:" << std::endl;
					for (std::size_t j = i; j < this->mK; ++j)
						out << "\t[" << j << "]";
					out << std::endl;

					for (std::size_t j = i; j < this->mK; ++j)
						out << "\t" << result[offset + j].index << ":" << result[offset + j].distance << " ";
					out << std::endl;

					for (std::size_t j = i; j < this->mK; ++j)
						out << "\t" << refResult[offset + j].index << ":" << refResult[offset + j].distance << " ";
					out << std::endl;
				}
			}
		}

		if (errors) {
			out << "Total errors: " << errors << std::endl;
		}
		else {
			out << "Verification OK." << std::endl;
		}

		refAlgorithm.cleanup();
		return errors == 0;
	}

	virtual void cleanup()
	{
		mResult.clear();
	}
};



/**
 * Interface (base class) for all embedding projection algorithms (second phase of ESOM).
 */
template <typename F = float>
class IProjectionAlgorithm
{
public:
	using TopkResult = typename TopkProblemInstance<F>::Result;

protected:
	std::size_t mDim, mN, mGridSize, mK;
	F mAdjust, mBoost;
	DataPoints<F> mResult;

public:
	IProjectionAlgorithm() : mDim(0), mN(0), mGridSize(0), mK(0), mAdjust(0), mBoost(0), mResult(2) {}

	virtual ~IProjectionAlgorithm() {}

	virtual void initialize(const DataPoints<F>& points, const DataPoints<F>& grid, const DataPoints<F>& grid2D,
							const std::vector<TopkResult>& neighbors, const std::size_t k, const F adjust, const F boost)
	{
		mDim = points.getDim();
		mN = points.size();

		if (grid.getDim() != mDim) {
			throw(bpp::RuntimeError() << "Data points and grid points have different dimensions: " << mDim << " != " << grid.getDim());
		}
		mGridSize = grid.size();

		if (grid2D.getDim() != 2) {
			throw(bpp::RuntimeError() << "Only 2D projection of the grid is currently supported (" << grid2D.getDim() << " dimensions given).");
		}
		if (grid2D.size() != mGridSize) {
			throw(bpp::RuntimeError() << "High-dimensional and low-dimensional grid coordinates have different sizes: " << mGridSize
									  << " != " << grid2D.size());
		}

		mK = k;
		mAdjust = adjust;
		mBoost = boost;

		// Derived class should save the pointers to inputs or copy them...
		// This part of the algorithm is not measured
	}

	virtual void prepareInputs()
	{
		mResult.resize(mN);
	}

	virtual void run() = 0;

	virtual const DataPoints<F>& getResults()
	{
		// Fetch the results (possibly from GPU memory).

		return mResult;
	}

	bool verifyResult(IProjectionAlgorithm<F>& refAlgorithm, std::ostream& out)
	{
		refAlgorithm.prepareInputs();
		refAlgorithm.run();
		auto refResult = refAlgorithm.getResults();
		auto result = getResults();

		std::size_t errors = 0;
		for (std::size_t p = 0; p < this->mN; ++p) {

			std::size_t i = 0;
			while (i < 2 && compare_floats_relative(refResult[p][i], result[p][i]) < 0.001)
				++i;

			if (i < 2) {
				if (++errors < 16) { // only first 16 errors are printed out
					out << "Error [" << p << "]:" << std::endl;
					for (std::size_t j = 0; j < 2; ++j)
						out << "\t[" << j << "]";
					out << std::endl;

					for (std::size_t j = 0; j < 2; ++j)
						out << "\t" << result[p][j];
					out << std::endl;

					for (std::size_t j = 0; j < 2; ++j)
						out << "\t" << refResult[p][j];
					out << std::endl;
				}
			}
		}

		if (errors) {
			out << "Total errors: " << errors << std::endl;
		}
		else {
			out << "Verification OK." << std::endl;
		}

		refAlgorithm.cleanup();
		return errors == 0;
	}

	virtual void cleanup()
	{
		mResult.clear();
	}
};

#endif
