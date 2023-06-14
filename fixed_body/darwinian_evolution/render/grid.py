class Grid:

	def __init__(self):
		self.min_x = None
		self.max_x = None
		self.min_y = None
		self.max_y = None
		self.width = None
		self.height = None
		self.core_position = None
		self.visited_coordinates = []

	BACK = 3
	FRONT = 0
	RIGHT = 1
	LEFT = 2

	# orientations from revolve1
	# SOUTH = 0 # Canvas.BACK
	# NORTH = 1 # Canvas.FRONT
	# EAST = 2 # Canvas.RIGHT
	# WEST = 3 # Canvas.LEFT

	# Current position of last drawn element
	x_pos = 0
	y_pos = 0

	# Orientation of robot
	
	orientation = FRONT

	# Direction of last movement
	previous_move = -1

	# Coordinates and orientation of movements
	movement_stack = [[0,0, FRONT]]

	def get_position(self):
		"""Return current position on x and y axis"""
		return [Grid.x_pos, Grid.y_pos]

	def set_position(self, x, y):
		"""Set position of x and y axis"""
		Grid.x_pos = x
		Grid.y_pos = y

	def set_orientation(self, orientation):
		"""Set new orientation on grid"""
		if orientation in [Grid.FRONT, Grid.RIGHT, Grid.BACK, Grid.LEFT]:
			Grid.orientation = orientation
		else:
			return False

	def calculate_orientation(self):
		"""Set orientation by previous move and orientation"""
		if (Grid.previous_move == -1 or
		(Grid.previous_move == Grid.FRONT and Grid.orientation == Grid.FRONT) or
		(Grid.previous_move == Grid.RIGHT and Grid.orientation == Grid.LEFT) or
		(Grid.previous_move == Grid.LEFT and Grid.orientation == Grid.RIGHT) or
		(Grid.previous_move == Grid.BACK and Grid.orientation == Grid.BACK)):
			self.set_orientation(Grid.FRONT)
		elif ((Grid.previous_move == Grid.RIGHT and Grid.orientation == Grid.FRONT) or
		(Grid.previous_move == Grid.BACK and Grid.orientation == Grid.LEFT) or
		(Grid.previous_move == Grid.FRONT and Grid.orientation == Grid.RIGHT) or
		(Grid.previous_move == Grid.LEFT and Grid.orientation == Grid.BACK)):
			self.set_orientation(Grid.RIGHT)
		elif ((Grid.previous_move == Grid.BACK and Grid.orientation == Grid.FRONT) or
		(Grid.previous_move == Grid.LEFT and Grid.orientation == Grid.LEFT) or
		(Grid.previous_move == Grid.RIGHT and Grid.orientation == Grid.RIGHT) or
		(Grid.previous_move == Grid.FRONT and Grid.orientation == Grid.BACK)):
			self.set_orientation(Grid.BACK)
		elif ((Grid.previous_move == Grid.LEFT and Grid.orientation == Grid.FRONT) or
		(Grid.previous_move == Grid.FRONT and Grid.orientation == Grid.LEFT) or
		(Grid.previous_move == Grid.BACK and Grid.orientation == Grid.RIGHT) or
		(Grid.previous_move == Grid.RIGHT and Grid.orientation == Grid.BACK)):
			self.set_orientation(Grid.LEFT)

	def move_by_slot(self, slot):
		"""Move in direction by slot id"""
		if slot == Grid.BACK:
			self.move_down()
		elif slot == Grid.FRONT:
			self.move_up()
		elif slot == Grid.RIGHT:
			self.move_right()
		elif slot == Grid.LEFT:
			self.move_left()


	def move_right(self):
		"""Set position one to the right in correct orientation"""
		if Grid.orientation == Grid.FRONT:
			Grid.x_pos += 1
		elif Grid.orientation == Grid.RIGHT:
			Grid.y_pos += 1
		elif Grid.orientation == Grid.BACK:
			Grid.x_pos -= 1
		elif Grid.orientation == Grid.LEFT:
			Grid.y_pos -= 1
		Grid.previous_move = Grid.RIGHT

	def move_left(self):
		"""Set position one to the left"""
		if Grid.orientation == Grid.FRONT:
			Grid.x_pos -= 1
		elif Grid.orientation == Grid.RIGHT:
			Grid.y_pos -= 1
		elif Grid.orientation == Grid.BACK:
			Grid.x_pos += 1
		elif Grid.orientation == Grid.LEFT:
			Grid.y_pos += 1
		Grid.previous_move = Grid.LEFT

	def move_up(self):
		"""Set position one upwards"""
		if Grid.orientation == Grid.FRONT:
			Grid.y_pos -= 1
		elif Grid.orientation == Grid.RIGHT:
			Grid.x_pos += 1
		elif Grid.orientation == Grid.BACK:
			Grid.y_pos += 1
		elif Grid.orientation == Grid.LEFT:
			Grid.x_pos -= 1
		Grid.previous_move = Grid.FRONT

	def move_down(self):
		"""Set position one downwards"""
		if Grid.orientation == Grid.FRONT:
			Grid.y_pos += 1
		elif Grid.orientation == Grid.RIGHT:
			Grid.x_pos -= 1
		elif Grid.orientation == Grid.BACK:
			Grid.y_pos -= 1
		elif Grid.orientation == Grid.LEFT:
			Grid.x_pos += 1
		Grid.previous_move = Grid.BACK

	def move_back(self):
		if len(Grid.movement_stack) > 1:
			Grid.movement_stack.pop()
		last_movement = Grid.movement_stack[-1]
		Grid.x_pos = last_movement[0]
		Grid.y_pos = last_movement[1]
		Grid.orientation = last_movement[2]

	def add_to_visited(self, include_sensors=True, is_sensor=False):
		"""Add current position to visited coordinates list"""
		self.calculate_orientation()
		if (include_sensors and is_sensor) or not is_sensor:
			self.visited_coordinates.append([Grid.x_pos, Grid.y_pos])
		Grid.movement_stack.append([Grid.x_pos, Grid.y_pos, Grid.orientation])

	def calculate_grid_dimensions(self):
		min_x = 0
		max_x = 0
		min_y = 0
		max_y = 0
		for coorinate in self.visited_coordinates:
			min_x = coorinate[0] if coorinate[0] < min_x else min_x
			max_x = coorinate[0] if coorinate[0] > max_x else max_x
			min_y = coorinate[1] if coorinate[1] < min_y else min_y
			max_y = coorinate[1] if coorinate[1] > max_y else max_y

		self.min_x = min_x
		self.max_x = max_x
		self.min_y = min_y
		self.max_y = max_y
		self.width = abs(min_x - max_x) + 1
		self.height = abs(min_y - max_y) + 1

	def calculate_core_position(self):
		self.core_position = [self.width - self.max_x - 1, self.height - self.max_y - 1]
		return self.core_position

	def reset_grid(self):
		Grid.x_pos = 0
		Grid.y_pos = 0
		Grid.orientation = Grid.FRONT
		Grid.previous_move = -1
		Grid.movement_stack = [[0,0, Grid.FRONT]]