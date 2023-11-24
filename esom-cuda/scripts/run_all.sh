#!/bin/bash

START_DATE=`date`

echo "Chill out first..."
sleep 10

./run_topk.sh
./run_projection.sh

END_DATE=`date`
echo "Done."
echo "Started at $START_DATE"
echo "Finished at $END_DATE"
