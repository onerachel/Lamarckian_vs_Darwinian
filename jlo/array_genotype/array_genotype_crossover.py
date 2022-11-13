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
              p: [],
              crossover_prob: ArrayCrossoverConfig):
    """
    To implement the uniform crossover, the following python code can be used.
    A uniform_crossover function is defined where incoming arguments A & B represent the parents,
    P denotes the probability matrix, and returning A & B represent the children.
    It can be observed that the information between parents is exchanged at the indexes where probability is less than the threshold (0.5) to form children.
    https://medium.com/@samiran.bera/crossover-operator-the-heart-of-genetic-algorithm-6c0fdcb405c0
    """

    p = np.random.rand(10)
    for i in range(len(p)):
        if p[i] < crossover_prob:
            temp = parent_a
            parent_a = parent_b
            parent_b = temp
    new_genotype = parent_b
    return new_genotype  # or parent_a as the new genotype
