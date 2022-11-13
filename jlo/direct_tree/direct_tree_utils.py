
from revolve2.core.modular_robot import Module
from typing import Iterable, Tuple, Optional, Union
from collections import deque
import queue
from revolve2.core.modular_robot import Brick,Core,ActiveHinge

def subtree_size(module: Module) -> int:
    """
    Calculates the size of the subtree starting from the module
    :param module: root of the subtree
    :return: how many modules the subtree has
    """
    count = 0
    if module is not None:
        for _ in bfs_iterate_modules(root=module):
            count += 1
    return count


def bfs_iterate_modules(root: Module,
                        include_none_child: bool = False) \
        -> Iterable[Tuple[Optional[Module], Module]]:
    """
    Iterates throw all modules breath first, yielding parent and current module
    :param root: root tree to iterate
    :return: iterator for all modules with respective parent in the form: `(Parent,Module)`
    """
    assert root is not None
    to_process = deque([(None, root)])
    while len(to_process) > 0:
        r: Tuple(Optional[Module], Module) = to_process.popleft()
        parent, elem = r
        if elem is not None:
            for child in elem.children:
                if child is not None or include_none_child:
                    to_process.append((elem, child))
        yield parent, elem

def recursive_iterate_modules(module: Module,
                              parent: Optional[Module] = None,
                              parent_slot: Optional[int] = None,
                              depth: int = 1,
                              include_none_child: bool = False) \
        -> Iterable[Tuple[Optional[Module], Optional[int], Module, int]]:
    """
    Iterate all modules, depth search first, yielding parent, parent slot, module and depth, starting from root_depth=1.
    Uses recursion.
    :param module: starting module to expand
    :param parent: for internal recursiveness, parent module. leave default
    :param parent_slot: for internal recursiveness, parent module slot. leave default
    :param depth: for internal recursiveness, depth of the module passed in. leave default
    :param include_none_child: if to include also None modules (consider empty as leaves)
    :return: iterator for all modules with (parent, parent_slot, module)
    """
    if module is not None:
        for slot, child in enumerate(module.children):
            if include_none_child or child is not None:
                for _next in recursive_iterate_modules(child, module, slot, depth+1):
                    yield _next
    yield parent, parent_slot, module

def duplicate_subtree(root: Module) -> Module:
    """
    Creates a duplicate of the subtree given as input
    :param root: root of the source subtree
    :return: new duplicated subtree
    """
    assert root is not None

    slot_queue = queue.Queue()  # infinite FIFO queue

    def append_new_empty_slots(module: Module):
        for slot, _ in enumerate(module.children):
            slot_queue.put((module, slot))

    dup_root = None

    # reset all module ids
    for parent, module in bfs_iterate_modules(root, include_none_child=True):
        new_module = None
        if module is not None:
            if isinstance(module, ActiveHinge):
                new_module = ActiveHinge(module.rotation)
            elif isinstance(module, Brick):
                new_module = Brick(module.rotation)
            elif isinstance(module, Core):
                new_module = Core(module.rotation)
            else:
                raise NotImplementedError()
            
            append_new_empty_slots(new_module)
            
        if parent is None:
            dup_root = new_module
        else:
            new_parent, slot = slot_queue.get_nowait()
            new_parent.children[slot] = new_module

        
            
    return dup_root

def clamp(value: Union[float, int],
          minvalue: Union[float, int],
          maxvalue: Union[float, int]) \
        -> Union[float, int]:
    """
    Clamps the value to a minimum and maximum
    :param value: source value
    :param minvalue: min possible value
    :param maxvalue: max possible value
    :return: clamped value
    """
    return min(max(minvalue, value), maxvalue)