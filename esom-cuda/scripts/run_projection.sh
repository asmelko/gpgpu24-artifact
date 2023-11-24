#!/bin/bash

START_DATE=`date`
. ./_run_config.sh

ALGO_TYPE='projection'
KS='7 15 31 63' # projection uses k+1

ALGOS='serial'
. ./_run_core.sh

ALGOS='cuda_base'
BLOCK_SIZES='8 16 32 64 128 256 512 1024'
. ./_run_core.sh

ALGOS='cuda_block_multi cuda_block_aligned cuda_block_aligned_register'
SHARED_MEM_SIZES='48k'
BLOCK_SIZES='8 16 32 64 128 256'
GROUPS_PER_BLOCKS='1 2 3 4'
. ./_run_core.sh

END_DATE=`date`
echo "Done."
echo "Started at $START_DATE"
echo "Finished at $END_DATE"
