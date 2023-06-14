class ArrayGenotypeConfig(object):
    pass


class RandomGenerateConfig:
    pass


class ArrayMutationConfig:
    def __init__(self,
                 mutation_prob,
                 genotype_conf):
        """
        Creates a MutationConfig object that sets the configuration for the mutation operator
        :param mutation_prob: mutation probability
        :param genotype_conf: configuration for the genotype to be mutated
        """
        self.mutation_prob = mutation_prob
        self.genotype_conf = genotype_conf


class ArrayCrossoverConfig:
    def __init__(self,
                 crossover_prob):
        """
        Creates a Crossover object that sets the configuration for the crossover operator
        :param crossover_prob: crossover probability
        """
        self.crossover_prob = crossover_prob
