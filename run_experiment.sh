#!/bin/bash

experiment_optimize=jlo/ea_fixedbody_cppn_openaies/optimize.py
experiment_name=openaies_1

screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}.log" nice -n19 python3 "${experiment_optimize}"


