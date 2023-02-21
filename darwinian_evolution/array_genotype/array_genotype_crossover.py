import random
import sys
import numpy as np
from typing import List, Tuple
#
from .array_genotype_config import ArrayGenotypeConfig, ArrayCrossoverConfig
from .array_genotype import ArrayGenotype
from revolve2.core.modular_robot import Module, Core


def crossover(parent_a: ArrayGenotype,
              parent_b: ArrayGenotype,
              crossover_prob: ArrayCrossoverConfig,
              first_best: bool):
    """
    To implement the uniform crossover, the following python code can be used.
    A uniform_crossover function is defined where incoming arguments A & B represent the parents,
    P denotes the probability matrix, and returning A & B represent the children.
    It can be observed that the information between parents is exchanged at the indexes where probability is less than the threshold (0.5) to form children.
    https://medium.com/@samiran.bera/crossover-operator-the-heart-of-genetic-algorithm-6c0fdcb405c0
    """

    prob_array = np.random.uniform(low=0, high=1.0, size=parent_a.params_array.shape[0])
    crossover_array = prob_array > crossover_prob
    new_params_array = np.copy(parent_a.params_array)
    new_params_array[crossover_array] = parent_b.params_array[crossover_array]
    
    new_genotype = ArrayGenotype(new_params_array)

    return new_genotype 
