#!/bin/bash

START_DATE=`date`
. ./_run_config.sh

ALGO_TYPE='topk'

ALGOS='serial'
. ./_run_core.sh

ALGOS='cuda_base'
BLOCK_SIZES='8 16 32 64 128 256'
. ./_run_core.sh

ALGOS='cuda_radixsort cuda_bitonicopt'
SHARED_MEM_SIZES='48k'
BLOCK_SIZES='32 64 128 256 512 1024'
. ./_run_core.sh

ALGOS='cuda_shm'
BLOCK_SIZES='64 128 256'
SHARED_MEM_SIZES='4k 8k 16k 32k 48k'
. ./_run_core.sh

SHARED_MEM_SIZES='48k'

ALGOS='cuda_2dinsert'
BLOCK_SIZES='16 32 64 128'
ITEMS_PER_BLOCKS='4 8 16 32'
ITEMS_PER_THREADS='1 2 3 4'
. ./_run_core.sh

ALGOS='cuda_2dinsert'
APPEND=1
BLOCK_SIZES='256'
ITEMS_PER_BLOCKS='8 16 32'
ITEMS_PER_THREADS='1 2 3 4'
. ./_run_core.sh

END_DATE=`date`
echo "Done."
echo "Started at $START_DATE"
echo "Finished at $END_DATE"
