#!/bin/bash

experiment_name=DRL+RevDE

for body in "babya" "babyb" "blokky" "garrix" "gecko" "insect" "linkin" "longleg" "penguin" "pentapod" "queen" "salamander" "squarish" "snake" "spider" "stingray" "tinlicker" "turtle" "ww" "zappa"

do
	for num in 1 2 3 4 5
	do
		screen -d -m -S "${experiment_name}" -L -Logfile "./${experiment_name}.log" nice -n19 python3 "optimize.py" $body $num &
#		python DRL/PPO/optimize.py $body $num &
	done
done
