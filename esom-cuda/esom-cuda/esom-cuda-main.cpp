#define _CRT_SECURE_NO_WARNINGS

#include "points.hpp"
#include "esom-serial.hpp"
#include "esom-cuda.hpp"
#include "topk-serial.hpp"
#include "topk-cuda.hpp"
#include "projection-serial.hpp"
#include "projection-cuda.hpp"
#include "structs.cuh"

#include "cli/args.hpp"
#include "system/stopwatch.hpp"
#include "cuda/cuda.hpp"

#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

class validation_fail_exception : public std::runtime_error
{
public:
	validation_fail_exception(const std::string& msg = "") : std::runtime_error(msg) {}
};

/**
 * Initialize the CUDA execution structure from arguments.
 */
void getCudaExecParameters(bpp::ProgramArguments& args, CudaExecParameters& exec)
{
	cudaDeviceProp props;
	CUCH(cudaGetDeviceProperties(&props, 0));

	exec.blockSize = (unsigned int)args.getArgInt("cudaBlockSize").getValue();
	if (exec.blockSize > (unsigned int)props.maxThreadsPerBlock)
		throw bpp::RuntimeError() << "Requested CUDA block size (" << exec.blockSize << ") exceeds device capabilities (" << props.maxThreadsPerBlock << ").";

	if (args.getArgInt("cudaSharedMemorySize").isPresent()) {
		exec.sharedMemorySize = (unsigned int)args.getArgInt("cudaSharedMemorySize").getValue();
		if (exec.sharedMemorySize > (unsigned int)props.sharedMemPerBlock)
			throw bpp::RuntimeError() << "Requested CUDA shared memory per block (" << exec.sharedMemorySize << ") exceeds device capabilities (" << props.sharedMemPerBlock << ").";
	}
	else
		exec.sharedMemorySize = (unsigned int)props.sharedMemPerBlock;

	exec.itemsPerBlock = args.getArgInt("itemsPerBlock").getAsUint32();
	exec.itemsPerThread = args.getArgInt("itemsPerThread").getAsUint32();
	exec.regsCache = args.getArgInt("regsCache").getAsUint32();
	exec.groupsPerBlock = args.getArgInt("groupsPerBlock").getAsUint32();

	std::cerr << "Cuda device #0 selected with CC " << props.major << "." << props.minor << " (" << exec.sharedMemorySize / 1024 << "kB shared mem)" << std::endl;
}


/**
 * Determine data dimension from arguments (and possibly by peeking the data files).
 */
std::size_t determineDim(bpp::ProgramArguments& args)
{
	std::size_t dim = 0;
	if (args.getArgInt("dim").isPresent()) {
		dim = args.getArgInt("dim").getValue();
	}

	std::size_t dataDim = 0;
	if (args.getArgBool("dataText").getValue()) {
		dataDim = peekTSVGetDimension(args.getArgString("data").getValue());
	}

	std::size_t gridDim = 0;
	if (args.getArgBool("gridText").getValue()) {
		gridDim = peekTSVGetDimension(args.getArgString("grid").getValue());
	}

	if (dim == 0 && (dataDim > 0 || gridDim > 0)) {
		// dimension is not set explicitly, but can be inferred from data files
		dim = (dataDim > 0) ? dataDim : gridDim;
	}

	if (dim == 0) {
		throw (bpp::RuntimeError() << "Unable to determine data dimension (-dim parameter required or input files must be text files).");
	}

	if (dataDim != 0 && dim != dataDim) {
		throw (bpp::RuntimeError() << "Input file data has dimension " << dataDim << " but " << dim << " dimension was set explicitly.");
	}

	if (gridDim != 0 && dim != gridDim) {
		throw (bpp::RuntimeError() << "Grid file data has dimension " << gridDim << " but " << dim << " dimension is expected.");
	}

	if (args.getArgBool("grid2DText").getValue() && args.getArgString("grid2D").isPresent()) {
		std::size_t grid2Dim = peekTSVGetDimension(args.getArgString("grid2D").getValue());
		if (grid2Dim != 2) {
			throw (bpp::RuntimeError() << "Low dim grid coordinates have dimension " << grid2Dim << " but only 2D projections are currently supported.");
		}
	}

	return dim;
}


/**
 * Load a datafile of points.
 * @param points data structure to be loaded
 * @param fileName where are the data stored
 * @param text true if the TSV format should be loaded, false for binary blob
 */
template<typename F = float>
void loadPointsFile(DataPoints<F> &points, const std::string& fileName, bool text = false)
{
	std::cerr << "Loading " << fileName << " points file";
	if (text) {
		std::cerr << " (in TSV format)";
	}
	std::cerr << " ... ";
	std::cerr.flush();
	
	if (text) {
		points.loadTSV(fileName);
	}
	else {
		points.loadBinary(fileName);
	}

	std::cerr << points.size() << " points, d=" << points.getDim() << std::endl;
}


/**
 * Generate regular 2D grid based on the size of the loaded (high-dimensional) grid.
 * @param grid to be generated
 * @param size total size of the grid (side^2)
 */
template<typename F = float>
void generateGrid(DataPoints<F>& grid, std::size_t size)
{
	std::size_t side = (std::size_t)round(sqrt((double)size));
	if (size != side * side) {
		throw(bpp::RuntimeError() << "Grid size " << size << " is not a square of an integer.");
	}

	// TODO - kontrola Kratochvil
	grid.resize(size);
	for (std::size_t y = 0; y < side; ++y) {
		F fy = (F)y * ((F)1.0 / (F)(side - 1));
		for (std::size_t x = 0; x < side; ++x) {
			F fx = (F)x * ((F)1.0 / (F)(side - 1));
			grid[y * side + x][0] = fx;
			grid[y * side + x][1] = fy;
		}
	}
}


/**
 * Generate top k(+1) grid neighbors of each point for the projection algorithm.
 * @param neighbors to be generated
 * @param points query points
 * @param grid reference points
 * @param k reference points
 */
template <typename F = float>
void generateNeighbors(std::vector<typename ITopkAlgorithm<F>::Result>& neighbors, std::size_t size, std::size_t gridSize, std::size_t k)
{
	k = k < gridSize ? k + 1 : k;

	neighbors.resize(size * k);

	for (std::size_t i = 0; i < size; ++i) {
		for (std::size_t j = 0; j < k; ++j) {
			auto offset = i * k + j;
			neighbors[offset].index = std::uint32_t((offset * 42) % gridSize);
			neighbors[offset].distance = F(offset) / gridSize;
		}
	}
}


/**
 * Find and return a an object representing top-k algorithm.
 */
template<typename F = float>
std::unique_ptr<ITopkAlgorithm<F>> getTopkAlgorithm(const std::string& name, CudaExecParameters& cudaExec)
{
	std::map<std::string, std::unique_ptr<ITopkAlgorithm<F>>> algorithms;
	
	// here we list all known topk algorithms (this list may extend)
	algorithms["serial"] = std::make_unique<TopkSerialAlgorithm<F>>();
	algorithms["cuda_base"] = std::make_unique<TopkCudaAlgorithm<F, TopkBaseKernel<F>>>(cudaExec);
	algorithms["cuda_shm"] = std::make_unique<TopkCudaAlgorithm<F, TopkThreadSharedKernel<F>>>(cudaExec);
	algorithms["cuda_radixsort"] = std::make_unique<TopkCudaAlgorithm<F, TopkBlockRadixSortKernel<F>>>(cudaExec);
	algorithms["cuda_2dinsert"] = std::make_unique<TopkCudaAlgorithm<F, Topk2DBlockInsertionSortKernel<F>>>(cudaExec);
	algorithms["cuda_bitonicsort"] = std::make_unique<TopkCudaAlgorithm<F, TopkBitonicSortKernel<F>>>(cudaExec);
	algorithms["cuda_bitonic"] = std::make_unique<TopkCudaAlgorithm<F, TopkBitonicKernel<F>>>(cudaExec);
	algorithms["cuda_bitonicopt"] = std::make_unique<TopkCudaAlgorithm<F, TopkBitonicOptKernel<F>>>(cudaExec);
	
	auto it = algorithms.find(name);
	if (it == algorithms.end()) {
		throw(bpp::RuntimeError() << "Unkown topk algorithm '" << name << "'.");
	}

	return std::move(it->second);
}


/**
 * Find and return a an object representing projection algorithm.
 */
template <typename F = float>
std::unique_ptr<IProjectionAlgorithm<F>> getProjectionAlgorithm(const std::string& name, CudaExecParameters& cudaExec)
{
	std::map<std::string, std::unique_ptr<IProjectionAlgorithm<F>>> algorithms;

	// here we list all known topk algorithms (this list may extend)
	algorithms["serial"] = std::make_unique<ProjectionSerialAlgorithm<F>>();
	algorithms["cuda_base"] = std::make_unique<ProjectionCudaAlgorithm<F, ProjectionBaseKernel<F>>>(cudaExec);
	algorithms["cuda_block"] = std::make_unique<ProjectionCudaAlgorithm<F, ProjectionBlockKernel<F>>>(cudaExec);
	algorithms["cuda_block_rectangle_index"] = std::make_unique<ProjectionCudaAlgorithm<F, ProjectionBlockRectangleIndexKernel<F>>>(cudaExec);
	algorithms["cuda_block_shared"] = std::make_unique<ProjectionCudaAlgorithm<F, ProjectionBlockSharedKernel<F>>>(cudaExec);
	algorithms["cuda_block_multi"] = std::make_unique<ProjectionCudaAlgorithm<F, ProjectionBlockMultiKernel<F>>>(cudaExec);
	algorithms["cuda_block_aligned"] = std::make_unique<AlignedProjectionCudaAlgorithm<F, ProjectionAlignedMemKernel<F>>>(cudaExec);
	algorithms["cuda_block_aligned_register"] = std::make_unique<AlignedProjectionCudaAlgorithm<F, ProjectionAlignedMemRegisterKernel<F>>>(cudaExec);
	algorithms["cuda_block_aligned_small"] = std::make_unique<AlignedProjectionCudaAlgorithm<F, ProjectionAlignedMemSmallKernel<F>>>(cudaExec);

	auto it = algorithms.find(name);
	if (it == algorithms.end()) {
		throw(bpp::RuntimeError() << "Unkown topk algorithm '" << name << "'.");
	}

	return std::move(it->second);
}


/**
 * Find and return a an object representing full ESOM algorithm.
 */
template<typename F = float>
std::unique_ptr<IEsomAlgorithm<F>> getEsomAlgorithm(const std::string& name, CudaExecParameters& cudaExec)
{
	std::map<std::string, std::unique_ptr<IEsomAlgorithm<F>>> algorithms;

	// here we list all known topk algorithms (this list may extend)
	algorithms["serial"] = std::make_unique<EsomSerialAlgorithm<F>>();
	algorithms["cuda_base"] = std::make_unique<EsomCudaAlgorithm<F, TopkBaseKernel<F>, ProjectionBaseKernel<F>>>(cudaExec);


	auto it = algorithms.find(name);
	if (it == algorithms.end()) {
		throw (bpp::RuntimeError() << "Unkown ESOM algorithm '" << name << "'.");
	}

	return std::move(it->second);
}


template<class ALG, bool ResultStorable = false>
void testAlgorithm(ALG* algorithm, ALG *refAlgorithm = nullptr, bpp::ProgramArguments* args = nullptr)
{
	{
		std::cerr << "Preparing ... ";
		std::cerr.flush();
		bpp::Stopwatch stopwatch(true);
		algorithm->prepareInputs();
		stopwatch.stop();
		std::cout << stopwatch.getMiliseconds() << ";";
		std::cout.flush();
		std::cerr << std::endl;
	}

	{
		std::cerr << "Executing ... ";
		std::cerr.flush();
		bpp::Stopwatch stopwatch(true);
		algorithm->run();
		stopwatch.stop();
		std::cout << stopwatch.getMiliseconds() << ";" << std::endl; // this is the endline also for CSV
	}

	if (refAlgorithm) {
		if (!algorithm->verifyResult(*refAlgorithm, std::cerr)) {
			throw validation_fail_exception();
		}
	}

	if constexpr (ResultStorable)
	{
		if (args && args->getArgString("out").isPresent())
		{
			auto fileName = args->getArgString("out").getValue();
			const auto& result = algorithm->getResults();

			if (args->getArgBool("outText").getValue()) {
				std::cerr << "Saving results in TSV format to " << fileName << " ..." << std::endl;
				result.saveTSV(fileName);			
			}
			else {
				std::cerr << "Saving results in binary format to " << fileName << " ..." << std::endl;
				result.saveBinary(fileName);
			}
		}
	}
	

	algorithm->cleanup();
}


template <typename F = float>
void printConfiguration(bpp::ProgramArguments& args, const DataPoints<F>& points, const DataPoints<F>& grid, const CudaExecParameters cudaExec)
{
	std::string algoType;
	if (args.getArg("topk").isPresent()) {
		algoType = "topk";
	}
	else if (args.getArg("projection").isPresent()) {
		algoType = "projection";
	}
	else {
		algoType = "esom";
	}
	std::string algoName = args.getArgString(algoType).getValue();

	// print selected algorithm
	std::cerr << "Problem type: ";
	std::cout << algoType << ";";
	std::cerr << std::endl;

	std::cerr << "Seleted algorithm: ";
	std::cout << algoName << ";";
	std::cerr << std::endl;

	// print parameters for the record
	std::cerr << "Problem configuration (n;dim;gridSize;k): ";
	std::cout << points.size() << ";" << points.getDim() << ";" << grid.size() << ";" << args.getArgInt("k").getValue() << ";";
	std::cerr << std::endl;

	// print CUDA execution parameters
	std::cerr << "CUDA configuration (blockSize,shmSize,groupsPerBlock,itemsPerBlock,itemsPerThres): ";
	std::cout << cudaExec.blockSize << ";" << cudaExec.sharedMemorySize << ";" << cudaExec.groupsPerBlock << ";" << cudaExec.itemsPerBlock << ";"
			  << cudaExec.itemsPerThread << ";";
	std::cerr << std::endl;
}


template<typename F = float>
void run(bpp::ProgramArguments& args)
{
	std::size_t dim = determineDim(args);
	std::size_t k = args.getArgInt("k").getValue();
	F smooth = (F)args.getArgFloat("smooth").getValue();
	F adjust = (F)args.getArgFloat("adjust").getValue();

	F boost = std::exp(-smooth - 1);
	if (boost < ProjectionProblemInstance<F>::minBoost)
		boost = ProjectionProblemInstance<F>::minBoost;

	std::string pointsFile = args.getArgString("data").getValue();
	std::string gridFile = args.getArgString("grid").getValue();

	DataPoints<F> points(dim);
	loadPointsFile(points, pointsFile, args.getArgBool("dataText").getValue());

	DataPoints<F> grid(dim);
	loadPointsFile(grid, gridFile, args.getArgBool("gridText").getValue());
	
	DataPoints<F> grid2d(2);
	if (args.getArgString("grid2D").isPresent()) {
		// load 2D grid from a file
		loadPointsFile(grid2d, args.getArgString("grid2D").getValue(), args.getArgBool("grid2DText").getValue());
	}
	else {
		// generate regular 2D grid in [0,1]^2
		generateGrid(grid2d, grid.size());
	}

	CudaExecParameters cudaExec;
	getCudaExecParameters(args, cudaExec);

	std::size_t repeat = args.getArgInt("repeat").getValue();
	if (repeat > 1) {
		std::cerr << "Repeating " << repeat << " times" << std::endl;
	}

	if (args.getArg("topk").isPresent()) {
		std::unique_ptr<ITopkAlgorithm<F>> algorithm = getTopkAlgorithm<F>(args.getArgString("topk").getValue(), cudaExec);
		algorithm->initialize(points, grid, k);
		std::unique_ptr<ITopkAlgorithm<F>> refAlgorithm;
		if (args.getArgBool("verify").getValue() && args.getArgString("topk").getValue() != "serial") {
			refAlgorithm = std::move(getTopkAlgorithm<F>("serial", cudaExec));
			refAlgorithm->initialize(points, grid, k);
		}

		for (std::size_t repetition = 1; repetition <= repeat; ++repetition) {
			if (repetition > 1) {
				std::cerr << std::endl << "Repeating exection for " << repetition << " time" << std::endl;
			}
			printConfiguration(args, points, grid, cudaExec);
			testAlgorithm(algorithm.get(), refAlgorithm.get(), &args);
		}
	}
	else if (args.getArg("projection").isPresent()) {
		std::vector<typename TopkProblemInstance<F>::Result> neighbors;
		generateNeighbors<F>(neighbors, points.size(), grid.size(), k);

		std::unique_ptr<IProjectionAlgorithm<F>> algorithm = getProjectionAlgorithm<F>(args.getArgString("projection").getValue(), cudaExec);
		algorithm->initialize(points, grid, grid2d, neighbors, k, adjust, boost);
		std::unique_ptr<IProjectionAlgorithm<F>> refAlgorithm;
		if (args.getArgBool("verify").getValue() && args.getArgString("projection").getValue() != "serial") {
			refAlgorithm = std::move(getProjectionAlgorithm<F>("serial", cudaExec));
			refAlgorithm->initialize(points, grid, grid2d, neighbors, k, adjust, boost);
		}

		for (std::size_t repetition = 1; repetition <= repeat; ++repetition) {
			if (repetition > 1) {
				std::cerr << std::endl << "Repeating exection for " << repetition << " time" << std::endl;
			}
			printConfiguration(args, points, grid, cudaExec);
			testAlgorithm<IProjectionAlgorithm<F>, true>(algorithm.get(), refAlgorithm.get(), &args);
		}
	}
	else { // full esom
		auto algorithm = getEsomAlgorithm<F>(args.getArgString("esom").getValue(), cudaExec);
		algorithm->initialize(points, grid, grid2d, k, adjust, boost);
		std::unique_ptr<IEsomAlgorithm<F>> refAlgorithm;
		if (args.getArgBool("verify").getValue() && args.getArgString("esom").getValue() != "serial") {
			refAlgorithm = std::move(getEsomAlgorithm<F>("serial", cudaExec));
			refAlgorithm->initialize(points, grid, grid2d, k, adjust, boost);
		}
		
		for (std::size_t repetition = 1; repetition <= repeat; ++repetition) {
			if (repetition > 1) {
				std::cerr << std::endl << "Repeating exection for " << repetition << " time" << std::endl;
			}
			printConfiguration(args, points, grid, cudaExec);
			testAlgorithm<IEsomAlgorithm<F>, true>(algorithm.get(), refAlgorithm.get(), &args);
		}
	}
}


int main(int argc, char* argv[])
{
	/*
	 * Arguments
	 */
	bpp::ProgramArguments args(0, 1);
	args.setNamelessCaption(0, "Output file (no output if missing).");

	try {
		args.registerArg<bpp::ProgramArguments::ArgInt>("dim", "Data dimensionality.", false, 2, 2);
		args.registerArg<bpp::ProgramArguments::ArgInt>("k", "How many nearest landmarks are used for every datapoint.", false, 16, 1);
		args.registerArg<bpp::ProgramArguments::ArgFloat>("smooth", "Produce smoother (positive values) or more rough approximation (negative values).", false, 1);
		args.registerArg<bpp::ProgramArguments::ArgFloat>("adjust", "How much non-local information to remove from the approximation.", false, 0);
		args.registerArg<bpp::ProgramArguments::ArgInt>("limit", "Maximal number of data points loaded from input file.", false, 1000000000, 0);
		args.registerArg<bpp::ProgramArguments::ArgInt>("repeat", "Repeat execution multiple times.", false, 1, 1);

		args.registerArg<bpp::ProgramArguments::ArgString>("data", "Data input file", true);
		args.registerArg<bpp::ProgramArguments::ArgString>("grid", "Grid coordinates in data dimension", true);
		args.registerArg<bpp::ProgramArguments::ArgString>("grid2D", "Grid coordinates in projection dimension (if missing, regular grid in unity square is created)", false);
		args.registerArg<bpp::ProgramArguments::ArgBool>("dataText", "Data input file is treated as TSV text file.");
		args.registerArg<bpp::ProgramArguments::ArgBool>("gridText", "Grid input file is treated as TSV text file.");
		args.registerArg<bpp::ProgramArguments::ArgBool>("grid2DText", "Grid 2D projection input file is treated as TSV text file.");
		args.registerArg<bpp::ProgramArguments::ArgEnum>("precision", "Specify single or double precision.", false, false, "single");
		args.getArgEnum("precision").addOptions({ "single", "double" });

		args.registerArg<bpp::ProgramArguments::ArgEnum>("esom", "Execute given ESOM algorithm.", false, true, "serial");
		args.getArgEnum("esom").addOptions({ "serial", "cuda_base" });
		args.registerArg<bpp::ProgramArguments::ArgEnum>("topk", "Execute first part of ESOM algorithm (topk).", false, true, "serial");
		args.getArgEnum("topk").addOptions(
			{ "serial", "cuda_base", "cuda_shm", "cuda_radixsort", "cuda_2dinsert", "cuda_bitonicsort", "cuda_bitonic", "cuda_bitonicopt" });
		args.registerArg<bpp::ProgramArguments::ArgEnum>("projection", "Execute second part of ESOM algorithm (aprox. projection).", false, true, "serial");
		args.getArgEnum("projection")
			.addOptions({ "serial", "cuda_base", "cuda_block", "cuda_block_rectangle_index", "cuda_block_shared", "cuda_block_multi",
						  "cuda_block_aligned", "cuda_block_aligned_register", "cuda_block_aligned_small" });

		args.registerArg<bpp::ProgramArguments::ArgString>("out", "Path to the output file where final map is stored.", false);
		args.registerArg<bpp::ProgramArguments::ArgBool>("outText", "Output is stored as TSV text file.");
		args.registerArg<bpp::ProgramArguments::ArgBool>("verify", "Whether the algorithm results should be verified against the serial algorithm.");

		// CUDA execution tuning parameters
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaBlockSize", "Number of threads in each block.", false, 256, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("cudaSharedMemorySize", "Amount of shared memory allocated per block.", false, 0, 0);
		args.registerArg<bpp::ProgramArguments::ArgInt>("itemsPerBlock", "Number of items associated with cuda block. Item defintion depends on selected algorithm.", false, 1, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("itemsPerThread", "Number of items associated with cuda block. Item defintion depends on selected algorithm.", false, 1, 1, 4096);
		args.registerArg<bpp::ProgramArguments::ArgInt>("groupsPerBlock", "Nubmer of thread groups of size cudaBlockSize that will reside in a block.", false, 1, 1);
		args.registerArg<bpp::ProgramArguments::ArgInt>("regsCache", "Size of the registry cache (extreme caching algorithms).", false, 1, 1);

		//args.registerArg<bpp::ProgramArguments::ArgFloat>("someConst", "Some tuning constant ....", false, 0.0, 0.0, 1.0);
		//args.registerArg<bpp::ProgramArguments::ArgBool>("randomize", "Shuffle the features (simulates random picking).");
		//args.registerArg<bpp::ProgramArguments::ArgInt>("randomSeed", "Seed used for the shuffle operation.", false, 42);
		//args.registerArg<bpp::ProgramArguments::ArgBool>("textIO", "Use text data files instead of binary data files.");

		// Process the arguments ...
		args.process(argc, argv);
	}
	catch (bpp::ArgumentException& e) {
		std::cout << "Invalid arguments: " << e.what() << std::endl << std::endl;
		args.printUsage(std::cout);
		return 1;
	}

	try {
		std::string precision = args.getArgString("precision").getValue();
		if (precision == "single")
			run<float>(args);
		else
			run<double>(args);
	}
	catch (std::exception& e) {
		std::cout << "Error: " << e.what() << std::endl << std::endl;
		return 2;
	}

	return 0;
}
