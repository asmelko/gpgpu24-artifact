#!/bin/bash

. ./_run_config.sh

RESULTS_DIR="./results/${HOSTNAME}/debug"
rm -rf "$RESULTS_DIR"

KS='16'
GRID_SIZES='256'
DIMS='32'

ALGO_TYPE='topk'
ALGOS='cuda_shm'
REPEAT=1
SHARED_MEM_SIZES='16k'
BLOCK_SIZES='512'
#EXTRA_ARGS='--verify'

EXEC="./esom-${HOSTNAME}"
#EXEC="./esom-${HOSTNAME}-old"
. ./_run_core.sh
