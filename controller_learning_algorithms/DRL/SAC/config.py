
# simulation parameters
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 5
NUM_ITERATIONS = 58
NUM_STEPS = 150
SIMULATION_TIME = 300

# loss weights
ACTOR_LOSS_COEFF = 1
CRITIC_LOSS_COEFF = 1

# learning rates
LR_ACTOR = 1e-3
LR_CRITIC = 3e-4
LR_ALPHA = 5e-3

# other parameters
GAMMA = 0.99

NUM_PARALLEL_AGENT = 10
BATCH_SIZE = int((NUM_PARALLEL_AGENT * NUM_STEPS) / 2)
N_EPOCHS = 20

# SAC parameters
INIT_TEMPERATURE = 0.1
BUFFER_SIZE = int((NUM_PARALLEL_AGENT * NUM_STEPS * 10))
TAU = 0.005

# number of past hinges positions to pass as observations
NUM_OBS_TIMES = 3

# dimension of the different types of  observations
NUM_OBSERVATIONS = 2

ACTION_CONSTRAINT = 1 #[-ACTION_CONSTRAINT, ACTION_CONSTRAINT]