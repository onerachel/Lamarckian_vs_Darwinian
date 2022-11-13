import random
import sys
from typing import List, Tuple

from .direct_tree_config import DirectTreeGenotypeConfig
from .direct_tree_utils import * 
from .direct_tree_genotype import DirectTreeGenotype
from revolve2.core.modular_robot import Module, Core

FRONT = 0
RIGHT = 1
BACK = 2
LEFT = 3

def crossover_list(parents: List[DirectTreeGenotype], conf: DirectTreeGenotypeConfig):
    assert len(parents) == 2
    return crossover(parents[0], parents[1], conf, None)


def crossover(parent_a: DirectTreeGenotype,
              parent_b: DirectTreeGenotype,
              conf: DirectTreeGenotypeConfig) \
        -> DirectTreeGenotype:
    """
    Performs actual crossover between two robot trees, parent_a and parent_b. This
    works as follows:
    - Robot `a` is copied
    - A random node `q` is picked from this copied robot. This may
      be anything but the root node.
    - We generate a list of nodes from robot b which, when replacing `q`,
      would not violate the robot's specifications. If this list is empty,
      crossover is not performed.
    - We pick a random node `r` from this list
    :return: New genotype
    """
    parent_a_size = subtree_size(parent_a.genotype.core)
    #parent_a_size = Analyzer(parent_a.genotype).num_modules
    genotype_child = parent_a.clone()
    empty_slot_list_a: List[Tuple[Module, int, Module]] = []  # parent, slot, child
    for _, _, module in recursive_iterate_modules(genotype_child.genotype.core, include_none_child=False):
        for slot, child in enumerate(module.children):
            empty_slot_list_a.append((module, slot, child))

    module_list_b: List[Tuple[Module, int]] = []
    for module_parent, _, module in recursive_iterate_modules(parent_b.genotype.core, include_none_child=False):
        if module_parent is None:
            continue
        module_size = subtree_size(module)
        module_list_b.append((module, module_size))

    crossover_point_found = False
    n_tries = 100
    while not crossover_point_found and n_tries > 0:
        n_tries -= 1
        module_parent_a, module_parent_a_slot, module_a = random.choice(empty_slot_list_a)
        module_a_size = subtree_size(module_a)

        def compatible(module_b: Module, module_b_size: int) -> bool:
            new_size = parent_a_size - module_a_size + module_b_size
            return conf.min_parts <= new_size <= conf.max_parts

        unrelated_module_list = [e for e in module_list_b if compatible(*e)]
        if not unrelated_module_list:
            continue

        module_b, _ = random.choice(unrelated_module_list)

        module_parent_a.children[module_parent_a_slot] = duplicate_subtree(module_b)
        crossover_point_found = True

    if not crossover_point_found:
        print(f'Crossover between genomes was not successful after 100 trials,'
              f' returning a clone of the first parent unchanged', file=sys.stderr)

    return genotype_child