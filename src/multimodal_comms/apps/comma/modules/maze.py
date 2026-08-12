import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
import random
import math


class MazePuzzle(Module):
    """The solver navigates a mouse through a maze to a colored sphere,
    pressing the correct button to disarm the module based on the maze’s layout.
    """
    def __init__(self, game_manager):
        super(MazePuzzle, self).__init__(game_manager)

        self.actions = {
            "move_up": self.move_up,
            "move_down": self.move_down,
            "move_left": self.move_left,
            "move_right": self.move_right,
            "check_position": self.check_position
        }
        
        if self.game_manager.config:
            config = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]
            self.maze = config["maze"]
            self.start_position = config["start_x"], config["start_y"]
        else:
            self.maze = [
                [1, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 2, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 0, 1],
                [1, 0, 1, 1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 1],
            ]
            self.start_position = self.get_random_start_position()
        
        self.move_buttons = []
        self.torus_colors = ["Green", "Blue", "Red", "Yellow"]
        self.sphere_colors = ["Blue", "Red", "Green", "Yellow"]
        self.accepting_positions = {'Green': (1, 1), 'Blue': (
            1, 7), 'Red': (7, 1), 'Yellow': (7, 7)}  # Adjusted positions
        
        self.current_position = self.start_position

        if self.game_manager.config:
            self.torus_color = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["torus_color"]
        else:
            self.torus_color = random.choice(self.torus_colors)
        
        self.accepting_color = self.get_accepting_color(self.torus_color)
        self.cell_size = 30  # Size of each cell in the maze
        self.x_offset = 70  # Initial x offset
        self.y_offset = 120  # Initial y offset
        self.maze_canvas_size = len(self.maze[0]) * self.cell_size
        self.create_widgets()
        self.draw_maze()
        self.update_mouse_position()

    def create_widgets(self):
        # Create the secondary canvas for the maze
        self.maze_canvas = tk.Canvas(
            self.canvas, width=self.maze_canvas_size, height=self.maze_canvas_size)
        self.canvas.create_window(
            (self.x_offset, self.y_offset), window=self.maze_canvas, anchor="nw")

    def draw_maze(self):
        self.maze_canvas.delete("all")
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                if cell == 1:
                    self.maze_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="black")
                elif cell == 2:
                    self.maze_canvas.create_oval(
                        x1, y1, x2, y2, fill=self.torus_color)
                else:
                    self.maze_canvas.create_rectangle(
                        x1, y1, x2, y2, fill="white")

        # Draw colored buttons at accepting positions
        for color, (x, y) in self.accepting_positions.items():
            self.maze_canvas.create_oval(y * self.cell_size + 10, x * self.cell_size + 10,
                                         y * self.cell_size + self.cell_size - 5, x * self.cell_size + self.cell_size - 5, fill=color, tags="oval")

    def get_random_start_position(self):
        while True:
            x = random.randint(0, len(self.maze) - 1)
            y = random.randint(0, len(self.maze[0]) - 1)
            if self.maze[x][y] == 0:
                return (x, y)

    def get_accepting_color(self, torus_color):
        index = self.torus_colors.index(torus_color)
        return self.sphere_colors[index]

    def move_up(self):
        self.move_mouse(-1, 0)

    def move_down(self):
        self.move_mouse(1, 0)

    def move_left(self):
        self.move_mouse(0, -1)

    def move_right(self):
        self.move_mouse(0, 1)

    def move_mouse(self, row_change, col_change):
        new_position = (
            self.current_position[0] + row_change, self.current_position[1] + col_change)
        if self.is_valid_move(new_position):
            self.current_position = new_position
            self.update_mouse_position()

    def is_valid_move(self, position):
        x, y = position
        if 0 <= x < len(self.maze) and 0 <= y < len(self.maze[0]):
            return self.maze[x][y] == 0
        return False

    def update_mouse_position(self):
        self.maze_canvas.delete("mouse")
        x, y = self.current_position[1] * \
            self.cell_size, self.current_position[0] * self.cell_size
        self.maze_canvas.create_oval(
            x + 10, y + 10, x + self.cell_size - 10, y + self.cell_size - 10, fill="grey", tags="mouse")

    def check_position(self):
        idx = self.torus_colors.index(self.torus_color)
        if self.current_position == self.accepting_positions[self.sphere_colors[idx]]:
            self.feedback = self.finish_check()
        else:
            self.log_mistake()
            self.reset_module()

    def reset_module(self):
        self.current_position = self.start_position
        self.update_mouse_position()

    def check_correct(self):
        return True

    def get_button_coordinates(self, button):
        return button.winfo_rootx() + 10, button.winfo_rooty() + 10

    def get_score(self):
        idx = self.torus_colors.index(self.torus_color)
        end = self.accepting_positions[self.sphere_colors[idx]]
        total = math.sqrt((end[0] - self.start_position[0])
                          ** 2 + (end[1] - self.start_position[1])**2)
        covered = math.sqrt((end[0] - self.current_position[0])
                            ** 2 + (end[1] - self.current_position[1])**2)
        if total == 0:
            return 0
        return abs(total - covered) / total

    def __str__(self):
        return "MazePuzzle"
