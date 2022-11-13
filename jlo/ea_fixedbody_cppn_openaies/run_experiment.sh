#!/bin/bash

set -e
set -x

experiment_optimize=optimize.py
experiment_name=NES_2
runs=2
runs_start=0

for i in $(seq $runs)
do
	run=$(($i+runs_start))
	screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}.log" nice -n15 python3 "${experiment_optimize}"
done
