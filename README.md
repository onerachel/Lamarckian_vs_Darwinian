# Lamarckian VS Darwinian
We compare Lamarckian evolution framework with Darwinian evolution framework in a mujoco and isaacgym based wrapper called Revolve2. The release version of Revolve2 used in this project is v0.3.6-beta1 (https://github.com/ci-group/revolve2/releases/tag/v0.3.6-beta1).

Both frameworks were tested with two tasks (panoramic rotation and point navigation). Parameters used in the experiments are:
pop_size=50,
offspring_size=25,
nr_generations=30,
learner==RevDE,
learning_trials=280,
simulation_time=30 secs for rotation task,
simulation_time=60 secs for point navigation task.

## Installation 
Steps to install:
``` 
1. Download isaacgym from https://developer.nvidia.com/isaac-gym
2. git clone git@github.com:onerachel/Lamarckian_vs_Darwinian.git
3. cd Lamarckian_vs_Darwinian
4. virtualenv -p=python3.8 .venv
5. source .venv/bin/activate
6. pip install ~/isaacgym/python/
7. ./dev_requirements.sh
``` 

## Run experiments 
To run experiments, e.g. darwinian_rotation and darwinian_point_navigation:
``` 
python darwinian_evolution/optimize.py
``` 
To show the simulation, add --visualize: 
``` 
python darwinian_evolution/optimize.py --visualize
``` 
To restart from the last optimization checkpoint, add --from_checkpoint: 
``` 
python darwinian_evolution/optimize.py --from_checkpoint
``` 
To plot fitness:
``` 
python darwinian_evolution/plot_fitness.py
``` 
To check the best robot wrt the fitnees:
``` 
cd darwinian_evoluation
python rerun_best.py
```
To check the best robot wrt the fitnees and save the video:
``` 
cd darwinian_evoluation
python rerun_best.py -r <OUTPUT-DIR>
```

## Examples


## Documentation 

[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/) 