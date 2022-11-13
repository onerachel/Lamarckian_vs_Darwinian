from revolve2.core.modular_robot import Core, Body, Module, Brick, ActiveHinge
from jlo.direct_tree.direct_tree_genotype import DirectTreeGenotype
import numpy as np
import math

"""
All the robots bodies are stored here
Use one of the functions to create your robot: ant,babya, babyb,blokky,garrix,gecko,insect,linkin,longleg,penguin,pentapod,queen,salamander,squarish,
snake,spider,stingray,tinlicker,turtle,ww,zappa,park,
"""


# def make_gecko_body() -> DirectTreeGenotype:
#     body = Body()
#     body.core.left = ActiveHinge(0.0)
#     body.core.left.attachment = Brick(0.0)
#     body.core.right = ActiveHinge(0.0)
#     body.core.right.attachment = Brick(0.0)
#     body.core.back = ActiveHinge(math.pi / 2)
#     body.core.back.attachment = Brick(math.pi / 2)
#     body.core.back.attachment.front = ActiveHinge(math.pi / 2)
#     body.core.back.attachment.front.attachment = Brick(math.pi / 2)
#     body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
#     body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
#     body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
#     body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)
#
#     return DirectTreeGenotype(body)

def make_super_ant_body() -> DirectTreeGenotype:
    body = Body()
    # head and first set of arms
    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = ActiveHinge(math.pi / 2)
    body.core.left.attachment.attachment = Brick(0.0)
    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = ActiveHinge(math.pi / 2)
    body.core.right.attachment.attachment = Brick(0.0)

    # second part of the body and arms
    body.core.back = ActiveHinge(math.pi / 2)
    body.core.back.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.left.attachment.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.right.attachment.attachment = Brick(0.0)

    # third part of the body and arms    
    body.core.back.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.left.attachment.attachment = Brick(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment = ActiveHinge(math.pi / 2)
    body.core.back.attachment.front.attachment.front.attachment.front.attachment.right.attachment.attachment = Brick(math.pi / 2)

    return DirectTreeGenotype(body)

def make_spider_body() -> DirectTreeGenotype:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = Body()


    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)

    body.core.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment = Brick(-np.pi / 2.0)
    body.core.front.attachment.front = ActiveHinge(0.0)
    body.core.front.attachment.front.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment = Brick(0.0)

    return DirectTreeGenotype(body)


def make_gecko_body() -> DirectTreeGenotype:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    return DirectTreeGenotype(body)


def make_babya_body() -> DirectTreeGenotype:
    """
    Get the babya modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.attachment.front.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    return DirectTreeGenotype(body)


def make_ant_body() -> DirectTreeGenotype:
    """
    Get the ant modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)

    body.core.right = ActiveHinge(0.0)
    body.core.right.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.right.attachment = Brick(0.0)

    body.core.back.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.attachment.right = ActiveHinge(0.0)
    body.core.back.attachment.front.attachment.right.attachment = Brick(0.0)

    return DirectTreeGenotype(body)


def make_salamander_body() -> DirectTreeGenotype:
    """
    Get the salamander modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = ActiveHinge(-np.pi / 2.0)

    body.core.right = ActiveHinge(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front = Brick(0.0)
    body.core.back.attachment.front.left = ActiveHinge(0.0)
    body.core.back.attachment.front.front = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.front.front.attachment = Brick(-np.pi / 2.0)

    body.core.back.attachment.front.front.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.front.front.attachment.left.attachment = Brick(0.0)
    body.core.back.attachment.front.front.attachment.left.attachment.left = Brick(0.0)
    body.core.back.attachment.front.front.attachment.left.attachment.front = (
        ActiveHinge(np.pi / 2.0)
    )
    body.core.back.attachment.front.front.attachment.left.attachment.front.attachment = ActiveHinge(
        -np.pi / 2.0
    )

    body.core.back.attachment.front.front.attachment.front = Brick(0.0)
    body.core.back.attachment.front.front.attachment.front.left = ActiveHinge(0.0)
    body.core.back.attachment.front.front.attachment.front.front = Brick(0.0)
    body.core.back.attachment.front.front.attachment.front.front.left = ActiveHinge(0.0)
    body.core.back.attachment.front.front.attachment.front.front.front = Brick(0.0)
    body.core.back.attachment.front.front.attachment.front.front.front.front = (
        ActiveHinge(np.pi / 2.0)
    )
    body.core.back.attachment.front.front.attachment.front.front.front.front.attachment = Brick(
        -np.pi / 2.0
    )
    body.core.back.attachment.front.front.attachment.front.front.front.front.attachment.left = Brick(
        0.0
    )
    body.core.back.attachment.front.front.attachment.front.front.front.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )

    return DirectTreeGenotype(body)


def make_blokky_body() -> DirectTreeGenotype:
    """
    Get the blokky modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.back = Brick(0.0)
    body.core.back.right = ActiveHinge(np.pi / 2.0)
    body.core.back.front = ActiveHinge(np.pi / 2.0)
    body.core.back.front.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.back.front.attachment.attachment = Brick(0.0)
    body.core.back.front.attachment.attachment.front = Brick(0.0)
    body.core.back.front.attachment.attachment.front.right = Brick(0.0)
    body.core.back.front.attachment.attachment.front.right.left = Brick(0.0)
    body.core.back.front.attachment.attachment.front.right.front = Brick(0.0)
    body.core.back.front.attachment.attachment.right = Brick(0.0)
    body.core.back.front.attachment.attachment.right.front = Brick(0.0)
    body.core.back.front.attachment.attachment.right.front.right = Brick(0.0)
    body.core.back.front.attachment.attachment.right.front.front = ActiveHinge(0.0)

    return DirectTreeGenotype(body)


def make_park_body() -> DirectTreeGenotype:
    """
    Get the park modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.back.attachment.attachment = Brick(0.0)
    body.core.back.attachment.attachment.right = Brick(0.0)
    body.core.back.attachment.attachment.left = ActiveHinge(0.0)
    body.core.back.attachment.attachment.front = Brick(0.0)
    body.core.back.attachment.attachment.front.right = ActiveHinge(-np.pi / 2.0)
    body.core.back.attachment.attachment.front.front = ActiveHinge(-np.pi / 2.0)
    body.core.back.attachment.attachment.front.left = ActiveHinge(0.0)
    body.core.back.attachment.attachment.front.left.attachment = Brick(0.0)
    body.core.back.attachment.attachment.front.left.attachment.right = ActiveHinge(
        -np.pi / 2.0
    )
    body.core.back.attachment.attachment.front.left.attachment.front = Brick(0.0)
    body.core.back.attachment.attachment.front.left.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.attachment.front.left.attachment.front.attachment = Brick(
        0.0
    )

    return DirectTreeGenotype(body)


def make_babyb_body() -> DirectTreeGenotype:
    """
    Get the babyb modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment = Brick(0.0)
    body.core.left.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment.front.attachment.front.attachment = Brick(0.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.front.attachment = Brick(0.0)
    body.core.right.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.front.attachment.front.attachment = Brick(0.0)

    body.core.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment = Brick(-np.pi / 2.0)
    body.core.front.attachment.front = ActiveHinge(0.0)
    body.core.front.attachment.front.attachment = Brick(0.0)
    body.core.front.attachment.front.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.front.attachment.front.attachment.front.attachment = Brick(0.0)

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment = Brick(-np.pi / 2.0)

    return DirectTreeGenotype(body)


def make_garrix_body() -> DirectTreeGenotype:
    """
    Get the garrix modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.front = ActiveHinge(np.pi / 2.0)

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = ActiveHinge(0.0)
    body.core.left.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.left.attachment.attachment.attachment = Brick(0.0)
    body.core.left.attachment.attachment.attachment.front = Brick(0.0)
    body.core.left.attachment.attachment.attachment.left = ActiveHinge(0.0)

    part2 = Brick(0.0)
    part2.right = ActiveHinge(np.pi / 2.0)
    part2.front = ActiveHinge(np.pi / 2.0)
    part2.left = ActiveHinge(0.0)
    part2.left.attachment = ActiveHinge(np.pi / 2.0)
    part2.left.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    part2.left.attachment.attachment.attachment = Brick(0.0)

    body.core.left.attachment.attachment.attachment.left.attachment = part2

    return DirectTreeGenotype(body)


def make_insect_body() -> DirectTreeGenotype:
    """
    Get the insect modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment = Brick(0.0)
    body.core.right.attachment.attachment.right = ActiveHinge(0.0)
    body.core.right.attachment.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.attachment.left = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.attachment.left.attachment = Brick(-np.pi / 2.0)
    body.core.right.attachment.attachment.left.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.right.attachment.attachment.left.attachment.right = ActiveHinge(0.0)
    body.core.right.attachment.attachment.left.attachment.right.attachment = (
        ActiveHinge(0.0)
    )
    body.core.right.attachment.attachment.left.attachment.right.attachment.attachment = ActiveHinge(
        np.pi / 2.0
    )

    return DirectTreeGenotype(body)


def make_linkin_body() -> DirectTreeGenotype:
    """
    Get the linkin modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment.attachment.attachment = Brick(0.0)

    part2 = body.core.right.attachment.attachment.attachment.attachment
    part2.front = Brick(0.0)

    part2.left = ActiveHinge(0.0)
    part2.left.attachment = ActiveHinge(0.0)

    part2.right = ActiveHinge(np.pi / 2.0)
    part2.right.attachment = ActiveHinge(-np.pi / 2.0)
    part2.right.attachment.attachment = ActiveHinge(0.0)
    part2.right.attachment.attachment.attachment = ActiveHinge(np.pi / 2.0)
    part2.right.attachment.attachment.attachment.attachment = ActiveHinge(0.0)

    return DirectTreeGenotype(body)


def make_longleg_body() -> DirectTreeGenotype:
    """
    Get the longleg modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment = ActiveHinge(0.0)
    body.core.left.attachment.attachment = ActiveHinge(0.0)
    body.core.left.attachment.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.left.attachment.attachment.attachment.attachment = ActiveHinge(0.0)
    body.core.left.attachment.attachment.attachment.attachment.attachment = Brick(0.0)

    part2 = body.core.left.attachment.attachment.attachment.attachment.attachment
    part2.right = ActiveHinge(0.0)
    part2.front = ActiveHinge(0.0)
    part2.left = ActiveHinge(np.pi / 2.0)
    part2.left.attachment = ActiveHinge(-np.pi / 2.0)
    part2.left.attachment.attachment = Brick(0.0)
    part2.left.attachment.attachment.right = ActiveHinge(np.pi / 2.0)
    part2.left.attachment.attachment.left = ActiveHinge(np.pi / 2.0)
    part2.left.attachment.attachment.left.attachment = ActiveHinge(0.0)

    return DirectTreeGenotype(body)


def make_penguin_body() -> DirectTreeGenotype:
    """
    Get the penguin modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.right = Brick(0.0)
    body.core.right.left = ActiveHinge(np.pi / 2.0)
    body.core.right.left.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.left.attachment.attachment = Brick(0.0)
    body.core.right.left.attachment.attachment.right = ActiveHinge(0.0)
    body.core.right.left.attachment.attachment.left = ActiveHinge(np.pi / 2.0)
    body.core.right.left.attachment.attachment.left.attachment = ActiveHinge(
        -np.pi / 2.0
    )
    body.core.right.left.attachment.attachment.left.attachment.attachment = ActiveHinge(
        np.pi / 2.0
    )
    body.core.right.left.attachment.attachment.left.attachment.attachment.attachment = (
        Brick(-np.pi / 2.0)
    )

    part2 = (
        body.core.right.left.attachment.attachment.left.attachment.attachment.attachment
    )

    part2.front = ActiveHinge(np.pi / 2.0)
    part2.front.attachment = Brick(-np.pi / 2.0)

    part2.right = ActiveHinge(0.0)
    part2.right.attachment = ActiveHinge(0.0)
    part2.right.attachment.attachment = ActiveHinge(np.pi / 2.0)
    part2.right.attachment.attachment.attachment = Brick(-np.pi / 2.0)

    part2.right.attachment.attachment.attachment.left = ActiveHinge(np.pi / 2.0)

    part2.right.attachment.attachment.attachment.right = Brick(0.0)
    part2.right.attachment.attachment.attachment.right.front = ActiveHinge(np.pi / 2.0)

    return DirectTreeGenotype(body)


def make_pentapod_body() -> DirectTreeGenotype:
    """
    Get the pentapod modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core.right.attachment.attachment.attachment.attachment

    part2.left = ActiveHinge(0.0)
    part2.front = ActiveHinge(np.pi / 2.0)
    part2.front.attachment = Brick(-np.pi / 2.0)
    part2.front.attachment.left = Brick(0.0)
    part2.front.attachment.right = ActiveHinge(0.0)
    part2.front.attachment.front = ActiveHinge(np.pi / 2.0)
    part2.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    part2.front.attachment.front.attachment.left = ActiveHinge(0.0)
    part2.front.attachment.front.attachment.right = ActiveHinge(0.0)

    return DirectTreeGenotype(body)


def make_queen_body() -> DirectTreeGenotype:
    """
    Get the queen modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core.right.attachment.attachment.attachment

    part2.left = ActiveHinge(0.0)
    part2.right = Brick(0.0)
    part2.right.front = Brick(0.0)
    part2.right.front.left = ActiveHinge(0.0)
    part2.right.front.right = ActiveHinge(0.0)

    part2.right.right = Brick(0.0)
    part2.right.right.front = ActiveHinge(np.pi / 2.0)
    part2.right.right.front.attachment = ActiveHinge(0.0)

    return DirectTreeGenotype(body)


def make_squarish_body() -> DirectTreeGenotype:
    """
    Get the squarish modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)
    body.core.back.attachment = Brick(0.0)
    body.core.back.attachment.front = ActiveHinge(0.0)
    body.core.back.attachment.left = ActiveHinge(np.pi / 2.0)
    body.core.back.attachment.left.attachment = Brick(-np.pi / 2.0)
    body.core.back.attachment.left.attachment.left = Brick(0.0)
    part2 = body.core.back.attachment.left.attachment.left

    part2.left = ActiveHinge(np.pi / 2.0)
    part2.front = ActiveHinge(0.0)
    part2.right = ActiveHinge(np.pi / 2.0)
    part2.right.attachment = Brick(-np.pi / 2.0)
    part2.right.attachment.left = Brick(0.0)
    part2.right.attachment.left.left = Brick(0.0)

    return DirectTreeGenotype(body)


def make_snake_body() -> DirectTreeGenotype:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = ActiveHinge(0.0)
    body.core.left.attachment = Brick(0.0)
    body.core.left.attachment.front = ActiveHinge(np.pi / 2.0)
    body.core.left.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core.left.attachment.front.attachment.front = ActiveHinge(0.0)
    body.core.left.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core.left.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        -np.pi / 2.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        0.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHinge(
        np.pi / 2.0
    )

    return DirectTreeGenotype(body)


def make_stingray_body() -> DirectTreeGenotype:
    """
    Get the stingray modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(np.pi / 2.0)
    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment = Brick(0.0)
    body.core.right.attachment.attachment.right = Brick(0.0)
    body.core.right.attachment.attachment.left = ActiveHinge(0.0)
    body.core.right.attachment.attachment.front = Brick(0.0)
    body.core.right.attachment.attachment.front.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.attachment.front.front = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment.attachment.front.left = ActiveHinge(0.0)
    body.core.right.attachment.attachment.front.left.attachment = Brick(0.0)
    body.core.right.attachment.attachment.front.left.attachment.right = ActiveHinge(
        np.pi / 2.0
    )
    body.core.right.attachment.attachment.front.left.attachment.front = ActiveHinge(0.0)
    body.core.right.attachment.attachment.front.left.attachment.front.attachment = (
        Brick(0.0)
    )

    return DirectTreeGenotype(body)


def make_tinlicker_body() -> DirectTreeGenotype:
    """
    Get the tinlicker modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core.right.attachment.attachment.attachment.attachment

    part2.left = Brick(0.0)
    part2.left.front = ActiveHinge(np.pi / 2.0)
    part2.left.right = Brick(0.0)
    part2.left.right.left = Brick(0.0)
    part2.left.right.front = ActiveHinge(0.0)
    part2.left.right.front.attachment = Brick(0.0)
    part2.left.right.front.attachment.front = ActiveHinge(np.pi / 2.0)
    part2.left.right.front.attachment.right = Brick(0.0)
    part2.left.right.front.attachment.right.right = ActiveHinge(np.pi / 2.0)
    return DirectTreeGenotype(body)


def make_turtle_body() -> DirectTreeGenotype:
    """
    Get the turtle modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.left = Brick(0.0)
    body.core.left.right = ActiveHinge(0.0)
    body.core.left.left = ActiveHinge(np.pi / 2.0)
    body.core.left.left.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.left.left.attachment.attachment = Brick(0.0)

    body.core.left.left.attachment.attachment.front = Brick(0.0)
    body.core.left.left.attachment.attachment.left = ActiveHinge(np.pi / 2.0)
    body.core.left.left.attachment.attachment.right = ActiveHinge(0.0)
    body.core.left.left.attachment.attachment.right.attachment = Brick(0.0)
    part2 = body.core.left.left.attachment.attachment.right.attachment

    part2.left = ActiveHinge(np.pi / 2.0)
    part2.left.attachment = ActiveHinge(-np.pi / 2.0)
    part2.front = Brick(0.0)
    part2.right = ActiveHinge(0.0)
    part2.right.attachment = Brick(0.0)
    part2.right.attachment.right = ActiveHinge(0.0)
    part2.right.attachment.left = ActiveHinge(np.pi / 2.0)
    part2.right.attachment.left.attachment = ActiveHinge(-np.pi / 2.0)
    part2.right.attachment.left.attachment.attachment = ActiveHinge(0.0)
    part2.right.attachment.left.attachment.attachment.attachment = ActiveHinge(0.0)

    return DirectTreeGenotype(body)


def make_ww_body() -> DirectTreeGenotype:
    """
    Get the ww modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)
    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment.attachment = Brick(0.0)
    body.core.right.attachment.attachment.attachment.left = ActiveHinge(0.0)
    body.core.right.attachment.attachment.attachment.left.attachment = Brick(0.0)
    part2 = body.core.right.attachment.attachment.attachment.left.attachment

    part2.left = ActiveHinge(0.0)
    part2.front = Brick(0.0)
    part2.front.right = ActiveHinge(np.pi / 2.0)
    part2.front.right.attachment = Brick(-np.pi / 2.0)
    part2.front.right.attachment.left = ActiveHinge(np.pi / 2.0)
    part2.front.right.attachment.left.attachment = ActiveHinge(0.0)
    part2.front.right.attachment.left.attachment.attachment = ActiveHinge(-np.pi / 2.0)

    return DirectTreeGenotype(body)


def make_zappa_body() -> DirectTreeGenotype:
    """
    Get the zappa modular robot.

    :returns: the robot.
    """
    body = Body()

    body.core.back = ActiveHinge(0.0)
    body.core.right = ActiveHinge(np.pi / 2.0)
    body.core.right.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment.attachment = ActiveHinge(-np.pi / 2.0)
    body.core.right.attachment.attachment.attachment.attachment = ActiveHinge(0.0)
    body.core.right.attachment.attachment.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core.right.attachment.attachment.attachment.attachment.attachment

    part2.front = ActiveHinge(0.0)
    part2.front.attachment = ActiveHinge(0.0)
    part2.left = ActiveHinge(np.pi / 2.0)
    part2.left.attachment = Brick(-np.pi / 2.0)
    part2.left.attachment.left = ActiveHinge(0.0)
    part2.left.attachment.left.attachment = Brick(0.0)
    part2.left.attachment.front = ActiveHinge(0.0)

    return DirectTreeGenotype(body)
