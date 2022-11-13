
import queue

from revolve2.core.modular_robot import Core, Body, Module, Brick, ActiveHinge
from jlo.direct_tree.direct_tree_config import DirectTreeGenotypeConfig
from jlo.direct_tree.direct_tree_genotype import DirectTreeGenotype
from typing import List, Optional, Type
import random
import math

possible_children: List[Optional[Type[Module]]] = [
    None,
    Brick,
    ActiveHinge
]

possible_orientation: List[float] = [0, math.pi/2, math.pi, math.pi*(3/2)]

def generate_tree(
                  max_parts: int,
                  n_parts_mu: float,
                  n_parts_sigma: float,
                  config: DirectTreeGenotypeConfig):
    """
    Generate
    :param max_parts: max number of blocks to generate
    :param n_parts_mu: target size of the tree (gauss mu)
    :param n_parts_sigma: variation of the target size of the tree (gauss sigma)
    :param config: genotype configuration
    :return: the new genotype
    """

    new_tree = Body()
    count_parts = 1
    _max_parts: int = math.floor(
        random.gauss(n_parts_mu, n_parts_sigma) + 0.5
    )
    max_parts = min(max_parts, _max_parts)

    children_probs: List[float] = [
        config.init.prob_no_child,
        config.init.prob_child_block,
        config.init.prob_child_active_joint,
    ]

    slot_queue = queue.Queue()  # infinite FIFO queue

    def append_new_empty_slots(module: Module):
        for slot, _ in enumerate(module.children):
            slot_queue.put((module, slot))

    append_new_empty_slots(new_tree.core)

    while count_parts <= max_parts and not slot_queue.empty():
        parent, slot = slot_queue.get_nowait()
        
        new_module: Module = generate_new_module(parent, slot, children_probs, config)
        if new_module is None:
            continue

        parent.children[slot] = new_module
        append_new_empty_slots(new_module)
        count_parts += 1
        
    return DirectTreeGenotype(new_tree)


def generate_new_module(parent,
    slot,
    children_probs,
    config,
    ) -> Optional[Module]:

    """
    Generates new random block
    :param parent: only for new module id
    :param slot: only for new module id
    :param children_probs: probabilites on how to select the new module, list of 3 floats.
    :param config: genotype configuration
    :return: returns new module, or None if the probability to not select any new module was extracted
    """

    new_child_module_class = random.choices(possible_children,
                                            weights=children_probs,
                                            k=1)[0]
    
    new_child_module = None
    
    if new_child_module_class != None:
        #random rotation
        rotation = random.choice(possible_orientation)       
        new_child_module = new_child_module_class(rotation)

    return new_child_module
    
    