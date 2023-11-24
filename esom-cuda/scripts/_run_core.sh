#!/bin/bash

# Main catesian loop that is used for every algorithm
# All used variables must be already set...

if [ ! -f "$EXEC" ]; then
	echo "Executable $EXEC is missing!";
	exit
fi

for ALGO in $ALGOS; do
for DIM in $DIMS; do
for GRID_SIZE in $GRID_SIZES; do
for K in $KS; do
	DIR="${RESULTS_DIR}/${ALGO_TYPE}"
	mkdir -p "$DIR"
	RES_FILE="$DIR/${ALGO}-${N}-${DIM}-${GRID_SIZE}-${K}.csv"
	
	if [ ! -f "$RES_FILE" -o "$APPEND" -ne 0 ]; then
		for BS in $BLOCK_SIZES; do
		for SHM in $SHARED_MEM_SIZES; do
		for GPB in $GROUPS_PER_BLOCKS; do
		for IPB in $ITEMS_PER_BLOCKS; do
		for IPT in $ITEMS_PER_THREADS; do
			$EXEC --${ALGO_TYPE} $ALGO --dim $DIM --k $K --data "${DATA_DIR}/points-1M-${DIM}.bin" --grid "${DATA_DIR}/grid-${GRID_SIZE}-${DIM}.bin" --grid2D "${DATA_DIR}/grid2d-${GRID_SIZE}.bin" --cudaBlockSize $BS --cudaSharedMemorySize $SHM --groupsPerBlock $GPB --itemsPerBlock $IPB --itemsPerThread $IPT --repeat $REPEAT $EXTRA_ARGS | tee -a "$RES_FILE"
			
			echo '------------------------' >&2
		done
		done
		done
		done
		done
	fi
done
done
done
done
