import cairo
import math
from .canvas import Canvas
from .grid import Grid
from revolve2.core.modular_robot import Core, ActiveHinge, Brick
import os

class Render:

    def __init__(self):
        """Instantiate grid"""
        self.grid = Grid()

    FRONT = 0
    BACK = 3
    RIGHT = 1
    LEFT = 2

    def parse_body_to_draw(self, canvas, module, slot, parent_rotation):
        """
        Parse the body to the canvas to draw the png
        @param canvas: instance of the Canvas class
        @param module: body of the robot
        @param slot: parent slot of module
        """
        #TODO: map slots to enumerators

        if isinstance(module, Core):
            canvas.draw_controller(module.id)
        elif isinstance(module, ActiveHinge):
            canvas.move_by_slot(slot)
            absolute_rotation = (parent_rotation + module.rotation) % math.pi
            Canvas.rotating_orientation = absolute_rotation
            canvas.draw_hinge(module.id)
            canvas.draw_connector_to_parent()
        elif isinstance(module, Brick):
            canvas.move_by_slot(slot)
            absolute_rotation = (parent_rotation + module.rotation) % math.pi
            Canvas.rotating_orientation = absolute_rotation
            canvas.draw_module(module.id)
            canvas.draw_connector_to_parent()

        # Traverse children of element to draw on canvas
        for core_slot, child_module in enumerate(module.children):
            if child_module is None:
                continue
            self.parse_body_to_draw(canvas, child_module, core_slot, module.rotation)
        canvas.move_back()

    def traverse_path_of_robot(self, module, slot, include_sensors=True):
        """
        Traverse path of robot to obtain visited coordinates
        @param module: body of the robot
        @param slot: attachment of parent slot
        @param include_sensors: add sensors to visisted_cooridnates if True
        """
        if isinstance(module, ActiveHinge) or isinstance(module, Brick):
            self.grid.move_by_slot(slot)
            self.grid.add_to_visited(include_sensors, False)
        # Traverse path of children of module
        for core_slot, child_module in enumerate(module.children):
            if child_module is None:
                continue
            self.traverse_path_of_robot(child_module, core_slot, include_sensors)
        self.grid.move_back()

    def render_robot(self, body, image_path):
        """
        Render robot and save image file
        @param body: body of robot
        @param image_path: file path for saving image
        """
        # Calculate dimensions of drawing and core position
        self.traverse_path_of_robot(body, Render.FRONT)
        self.grid.calculate_grid_dimensions()
        core_position = self.grid.calculate_core_position()

        # Draw canvas
        cv = Canvas(self.grid.width, self.grid.height, 100)
        cv.set_position(core_position[0], core_position[1])

        # Draw body of robot
        self.parse_body_to_draw(cv, body, Render.FRONT, 0)

        # Draw sensors after, so that they don't get overdrawn
        #cv.draw_sensors()
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv.save_png(image_path)

        # Reset variables to default values
        cv.reset_canvas()
        self.grid.reset_grid()