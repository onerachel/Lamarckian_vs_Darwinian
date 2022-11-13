
# simulation parameters
SAMPLING_FREQUENCY = 5
CONTROL_FREQUENCY = 5
NUM_ITERATIONS = 58  #episode: the number of batches needed to complete one epoch
NUM_STEPS = 150
SIMULATION_TIME = 30

# PPO parameters
PPO_CLIP_EPS = 0.2
PPO_LAMBDA = 0.95

# loss weights
ACTOR_LOSS_COEFF = 1
CRITIC_LOSS_COEFF = 1
ENTROPY_COEFF = 0.01

# learning rates
LR_ACTOR = 1e-2
LR_CRITIC = 5e-4

# other parameters
GAMMA = 0.99

NUM_PARALLEL_AGENT = 10
BATCH_SIZE = int((NUM_PARALLEL_AGENT*NUM_STEPS) / 2)
N_EPOCHS = 4

# number of past hinges positions to pass as observations
NUM_OBS_TIMES = 3

# dimension of the different types of  observations
NUM_OBSERVATIONS = 2

ACTION_CONSTRAINT = 1 #[-ACTION_CONSTRAINT, ACTION_CONSTRAINT]