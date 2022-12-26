

from revolve2.core.modular_robot import Core, Module, ActiveHinge

from .random_tree_generator import generate_new_module
from .direct_tree_config import DirectTreeGenotypeConfig
from .direct_tree_utils import *
from .direct_tree_genotype import DirectTreeGenotype
from revolve2.core.modular_robot import Body

from typing import Optional, Tuple, List
import random

FRONT = 0
RIGHT = 1
BACK = 2
LEFT = 3


def decide(
    probability: float,
    rng: random.Random
) -> bool:
    return rng.random() < probability

def mutate(
    genotype: DirectTreeGenotype,
    config: DirectTreeGenotypeConfig,
    rng: random.Random
) -> DirectTreeGenotype:
    """
    Mutates the robot tree. This performs the following operations:
    - Body parameters are mutated
    - Brain parameters are mutated
    - A subtree might be removed
    - A subtree might be duplicated
    - Two subtrees might be swapped
    - Subtrees are duplicated at random
    - Body parts are added at random
    :param genotype:
    :param config:
    :return: mutated version of the genome
    """

    body = Body()
    body.core = duplicate_subtree(genotype.genotype.core)
    new_genotype = DirectTreeGenotype(body)

    # delete_random_subtree
    if decide(config.mutation.p_delete_subtree, rng):
        r, n = delete_random_subtree(new_genotype, config)

    # generate random new module
    if decide(config.mutation.p_generate_subtree, rng):
        r = generate_random_new_module(new_genotype, config)

    # TODO random rotate modules

    # duplicate random subtree
    #duplicazione elementi e non dei riferimenti TODO
    if decide(config.mutation.p_duplicate_subtree, rng):
        duplicate_random_subtree(new_genotype, config)

    # swap random subtree
    if decide(config.mutation.p_swap_subtree, rng):
        swap_random_subtree(new_genotype)

    # mutate oscillators TODO
    #if decide(config.mutation.p_mutate_oscillators, rng):
        # mutate_oscillators(genotype, config, rng)

    return new_genotype


def delete_random_subtree(genotype: DirectTreeGenotype,
                          genotype_conf: DirectTreeGenotypeConfig) -> Tuple[Optional[Module], int]:
    """
    Deletes a subtree at random, assuming this is possible within
    the boundaries of the robot specification.
    :param root: Root node of the tree
    :param genotype_conf:
    :return: The removed subtree (or None if no subtree was removed) and the size of the subtree (the amount of modules removed)
    """

    #analyzer = Analyzer(genotype.genotype)
    robot_size = subtree_size(genotype.genotype.core)
    max_remove_list = robot_size - genotype_conf.min_parts

    module_list = []
    for parent, parent_slot, module in recursive_iterate_modules(genotype.genotype.core):
        if parent is not None:
            _subtree_size = subtree_size(module)
            if _subtree_size <= max_remove_list:
                module_list.append((parent, module, _subtree_size))

    if not module_list:
        return None, 0

    parent, subtree_root, _subtree_size = random.choice(module_list)
    for i, module in enumerate(parent.children):
        if module == subtree_root:
            parent.children[i] = None
            break
    else:
        # break was not reached, module not found about children
        raise RuntimeError("Subtree not found in the parent module!")
    return subtree_root, _subtree_size


def generate_random_new_module(genotype: DirectTreeGenotype,
                               genotype_conf: DirectTreeGenotypeConfig) \
        -> Optional[Module]:
    """
    Generates a new random module at a random position. It fails if the robot is already too big.
    Could generate an invalid robot.
    :param root: root of the robot tree
    :param genotype_conf: genotype configuration
    :return: reference to the new module, None if no insertion was possible
    """
    #analyzer = Analyzer(genotype.genotype)
    robotsize: int = subtree_size(genotype.genotype.core)
    if genotype_conf.max_parts == robotsize:
        return None

    empty_slot_list: List[Tuple[Module, int]] = []
    for parent, parent_slot, module in recursive_iterate_modules(genotype.genotype.core):
        # Create empty slot list
        for slot, child in enumerate(module.children):
            if child is None:
                empty_slot_list.append((module, slot))

    if not empty_slot_list:
        return None

    # choose random empty slot to where the duplication is created
    target_parent, target_empty_slot = random.choice(empty_slot_list) 

    possible_children_probs: List[float] = [
        0,
        genotype_conf.init.prob_child_block,
        genotype_conf.init.prob_child_active_joint,
    ]

    new_module = generate_new_module(target_parent, target_empty_slot, possible_children_probs, genotype_conf)

    if new_module is None:
        # randomly chose to close this slot, not enabled
        return None

    target_parent.children[target_empty_slot] = new_module




def duplicate_random_subtree(genotype: DirectTreeGenotype, conf: DirectTreeGenotypeConfig) -> bool:
    """
    Picks a random subtree that can be duplicated within the robot
    boundaries, copies it and attaches it to a random free slot.
    :param root: root of the robot tree
    :param conf: direct tree genotype configuration
    :return: True if duplication happened
    """
    robotsize = subtree_size(genotype.genotype.core)
    max_add_size = conf.max_parts - robotsize

    # Create a list of subtrees that is not larger than max_add_size
    module_list: List[Tuple[Module, Module, int]] = []
    empty_slot_list: List[Tuple[Module, int]] = []
    for parent, parent_slot, module in recursive_iterate_modules(genotype.genotype.core):
        # Create empty slot list
        for slot, child in enumerate(module.children):
            if child is None:
                empty_slot_list.append((module, slot))

        if parent is None:
            continue

        _subtree_size = subtree_size(module)
        # This line of code above it's slow, because it's recalculated for each subtree.
        # But I don't care at the moment. You can speed it up if you want.
        if _subtree_size > max_add_size:
            continue
        # Create possible source subtree list
        module_list.append((parent, module, _subtree_size))

    if not module_list:
        return False
    if not empty_slot_list:
        return False

    # choose random tree to duplicate
    parent, subtree_root, _subtree_size = random.choice(module_list)
    # choose random empty slot to where the duplication is created
    target_parent, target_empty_slot = random.choice(empty_slot_list)

    # deep copy the source subtree
    subtree_root = duplicate_subtree(subtree_root)
    # and attach it
    target_parent.children[target_empty_slot] = subtree_root

    return True


def swap_random_subtree(genotype: DirectTreeGenotype) -> bool:
    """
    Picks two random subtrees (which are not parents / children of each
    other) and swaps them.
    :param root: root of the robot tree
    :return: True if swapping happened
    """
    module_list: List[Tuple[Module, int, Module]] = []
    for parent, parent_slot, module in recursive_iterate_modules(genotype.genotype.core, include_none_child=True):
        if parent is None:
            continue
        module_list.append((parent, parent_slot, module))

    parent_a, parent_a_slot, a = random.choice(module_list)
    a_module_set = set()
    for _, _, module in recursive_iterate_modules(a):
        a_module_set.add(module)

    unrelated_module_list = [e for e in module_list if e[2] not in a_module_set]
    if not unrelated_module_list:
        return False

    parent_b, parent_b_slot, b = random.choice(unrelated_module_list)

    parent_b.children[parent_b_slot] = a
    parent_a.children[parent_a_slot] = b

    return True

def mutate_oscillators(genotype: DirectTreeGenotype, conf: DirectTreeGenotypeConfig, rng) -> None:
    """
    Mutates oscillation
    :param root: root of the robot tree
    :param conf: genotype config for mutation probabilities
    """

    for _, _, module in recursive_iterate_modules(genotype.genotype.core):
        if isinstance(module, ActiveHinge):
            if decide(conf.mutation.p_mutate_oscillator, rng):
                module.oscillator_amplitude += random.gauss(0, conf.mutation.mutate_oscillator_amplitude_sigma)
                module.oscillator_period += random.gauss(0, conf.mutation.mutate_oscillator_period_sigma)
                module.oscillator_phase += random.gauss(0, conf.mutation.mutate_oscillator_phase_sigma)

                # amplitude is clamped between 0 and 1
                module.oscillator_amplitude = clamp(module.oscillator_amplitude, 0, 1)
                # phase and period are periodically repeating every max_oscillation,
                #  so we bound the value between [0,conf.max_oscillation] for convenience
                module.oscillator_phase = module.oscillator_phase % conf.max_oscillation
                module.oscillator_period = module.oscillator_period % conf.max_oscillation
