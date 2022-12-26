#!/bin/bash

experiment_optimize=darwinian_evolution/optimize.py
experiment_name=darwinian_rotation

for num in {1..2}
do
  screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}.log" nice -n19 python3 "${experiment_optimize}"
done
