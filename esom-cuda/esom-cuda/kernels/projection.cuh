#ifndef ESOM_CUDA_PROJECTION_CUH
#define ESOM_CUDA_PROJECTION_CUH

#include <cstddef>
#include <cstdint>

#include "structs.cuh"


#define DECLARE_PROJECTION_KERNEL(NAME)                                                                                                          \
template <typename F>                                                                                                                            \
class NAME                                                                                                                                       \
{                                                                                                                                                \
public:                                                                                                                                          \
	static void run(const ProjectionProblemInstance<F>& in, CudaExecParameters& exec);                                                           \
};

DECLARE_PROJECTION_KERNEL(ProjectionBaseKernel)
DECLARE_PROJECTION_KERNEL(ProjectionBlockKernel)
DECLARE_PROJECTION_KERNEL(ProjectionBlockSharedKernel)
DECLARE_PROJECTION_KERNEL(ProjectionBlockRectangleIndexKernel)
DECLARE_PROJECTION_KERNEL(ProjectionBlockMultiKernel)
DECLARE_PROJECTION_KERNEL(ProjectionAlignedMemKernel)
DECLARE_PROJECTION_KERNEL(ProjectionAlignedMemRegisterKernel)
DECLARE_PROJECTION_KERNEL(ProjectionAlignedMemSmallKernel)

#endif
