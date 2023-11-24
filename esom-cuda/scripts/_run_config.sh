#!/bin/bash

HOSTNAME=`hostname -s`
EXEC="./esom-${HOSTNAME}"
DATA_DIR='./data'
RESULTS_DIR="./results/${HOSTNAME}"

# How many times is each test repeated
REPEAT=10

# These are just placeholders which needs to be overriden
ALGO_TYPE='topk'
ALGOS='cuda_bitonicopt'

# Basic problem configuration parameters (also projected into result file names)
N='1M'
DIMS='4 8 16 32 64'
KS='8 16 32 64'
GRID_SIZES='128 256 512 1024'

# Tuning parameters (defaults)
BLOCK_SIZES='256'
SHARED_MEM_SIZES='0'
GROUPS_PER_BLOCKS='1'
ITEMS_PER_BLOCKS='1'
ITEMS_PER_THREADS='1'

# In case we need something really special
EXTRA_ARGS=''

# If set to 1, the core loop will ignore that file exists and append to it anyways
APPEND=0
