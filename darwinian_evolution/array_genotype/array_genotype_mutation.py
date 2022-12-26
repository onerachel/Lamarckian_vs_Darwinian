from __future__ import division
import math
import random

from .array_genotype import ArrayGenotype
from .array_genotype_config import ArrayMutationConfig

from itertools import repeat

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

######################################
# GA Mutations                       #
######################################

def mutate(genotype: ArrayGenotype, mu, sigma, mutation_prob: ArrayMutationConfig):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input genotype (brain, array of weights). This mutation expects a
    :term:`sequence` individual composed of vectors.
    The *mutation_prob* argument is the probability of each vector to be mutated.
    :param genotype: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param mutation_prob: Independent probability for each attribute to be mutated.
    :returns: new genotype.
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py
    """
    size = len(genotype.genotype)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

    for i, m, s in zip(range(size), mu, sigma):
        if random.random() < mutation_prob:
            genotype.genotype[i] += random.gauss(m, s)

    return genotype  # new_genotype