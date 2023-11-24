# Artifact Submission: GPU EmbedSOM

This repository contains the optimized GPU implementation of EmbedSOM together with the empirically measured data and generated plots.

## Overview

The artifact comprises the following directories:

* `data` -- CSV data files with measurements from our GPU cluster
* `plots` -- plots generated by the data from `data` directory (including those that were not published in the paper due to the page limit)
* `esom-cuda` -- the CUDA implementation of EmbedSOM


## Detailed artifact contents

`esom-cuda/esom-cuda/kernels` directory contains the source files to the CUDA kernels. The names of kernels in this directory are in the original form. In the paper, we decided to rename them for the sake of better readability and simpler understanding. Here we provide the translation table for the kernel names:

| Optimization name in the paper | Kernel name in the source files |
| --------------------------- | ----------- |
| **Shared** kNN  | `topkThreadSharedKernel` |
| **GridInsert** kNN | `topk2DBlockInsertionSortKernel` |
| **Bitonic** kNN | `topkBitonicOptKernel` |
| **Shared** projection |  `projectionBlockMultiKernel` |
| **Aligned** projection | `projectionBlockMultiAlignedMemKernel`|
| **Registers** projection | `projectionBlockMultiAlignedMemRegisterKernel` |


## Measured results

The results were measured using `run_all.sh` script in `esom-cuda/scripts` on our GPU cluster. The measurements are stored in the `data` directory as CSV files with self-explanatory headers.

The `plots` directory contains figures generated from the measurement data. Here we describe each figure in the directory:

| Plot | Description | Figure number in paper |
| --------------------------- | ----------- | -- |
| `alg_knn_repre_ampere.pdf`| kNN kernels comparison on realistic values of `d` and `k` parameters | Figure 3
| `alg_knn_extreme_ampere.pdf`| kNN kernels comparison on extreme values of `d` and `k` parameters | not included
| `alg_proj_repre_ampere.pdf`| projection kernels comparison on realistic values of `d` and `k` parameters | Figure 4
| `alg_proj_extreme_ampere.pdf`| projection kernels comparison on extreme values of `d` and `k` parameters | not included
| `esom_percent_ampere.pdf`| the relative time spent in kNN kernel | Figure 5
| `esom_repre_ampere.pdf`| EmbedSOM implementations comparison on realistic values of `d` and `k` parameters | not included
| `esom_extreme_ampere.pdf`| EmbedSOM implementations comparison on extreme values of `d` and `k` parameters | not included