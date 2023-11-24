#ifndef ESOM_CUDA_TOPK_CUH
#define ESOM_CUDA_TOPK_CUH

#include "structs.cuh"

#include <cstddef>
#include <cstdint>


#define DECLARE_TOPK_KERNEL(NAME)													\
template<typename F>																\
class NAME																			\
{																					\
public:																				\
	static void run(const TopkProblemInstance<F>& in, CudaExecParameters& exec);	\
};

DECLARE_TOPK_KERNEL(TopkBaseKernel)
DECLARE_TOPK_KERNEL(TopkThreadSharedKernel)
DECLARE_TOPK_KERNEL(TopkBlockRadixSortKernel)
DECLARE_TOPK_KERNEL(Topk2DBlockInsertionSortKernel)
DECLARE_TOPK_KERNEL(TopkBitonicSortKernel)
DECLARE_TOPK_KERNEL(TopkBitonicKernel)
DECLARE_TOPK_KERNEL(TopkBitonicOptKernel)

#endif
