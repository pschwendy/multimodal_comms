import numpy as np
import random
import functools

from collections import Counter
from multimodal_comms.apps.comma.modules.module import Module
from multimodal_comms.apps.comma.modules.util import HighlightButton

class SimpleWirePuzzle(Module):
    """Wire puzzle closely following the setting in Keep Talking and Nobody Explodes
    """
    def __init__(self, game_manager):
        super(SimpleWirePuzzle, self).__init__(game_manager)

        self.actions = {}
        
        # Config
        min_wires = 3
        max_wires = 6
        
        if self.game_manager.config:
            self.n_wires = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["n_wires"]
        else:
            self.n_wires = random.randint(min_wires, max_wires)

        colors = ["red", "white", "yellow", "blue", "black"]

        self.cut_button_width = 50
        self.cut_button_height = 40

        right_top = 205
        left_top = 25
        
        self.canvas.create_rectangle(left_top, 25, 50, self.height - 25, outline="black", width=2)
        self.canvas.create_rectangle(self.width - 50, right_top, self.width - 25, self.height - 25, outline="black", width=2)

        wire_width = 10
        pad = 30
        wire_start_left = np.linspace(left_top + pad, self.height - 25 - pad - wire_width, self.n_wires)
        wire_start_right = np.linspace(right_top + pad,  self.height - 25 - pad - wire_width, self.n_wires)

        self.wire_colors = []
        self.buttons = []

        color_list = []
        if self.game_manager.config:
            color_list = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["colors"]
        else:
            color_list = [random.choice(colors) for _ in range(self.n_wires)]

        for i, start_y in enumerate(wire_start_left):
            color = color_list[i]
            self.canvas.create_polygon(50, start_y, self.width - 50, wire_start_right[i], self.width - 50, wire_start_right[i] + wire_width, 50, start_y + wire_width, fill=color)
            button = HighlightButton(self, (self.width - 25) / 2, (wire_start_right[i] + start_y + wire_width) / 2 - self.cut_button_height / 2, self.cut_button_width, self.cut_button_height, text="CUT", font=("Helvetica", 12), command=functools.partial(self.game_manager.solver.step, f"cut_wire_{i + 1}"))
            self.buttons.append(button)
            self.wire_colors.append(color)
            self.actions[f"cut_wire_{i + 1}"] = functools.partial(self.on_wire_cut, None, i)


    def check_correct(self):
        wire_counts = Counter(self.wire_colors)
        if self.n_wires == 3:
            if "red" not in wire_counts:
                answer = 1
            elif self.wire_colors[-1] == "white":
                answer = 2
            elif "blue" in wire_counts and wire_counts["blue"] > 1:
                last_blue = 0
                for i in range(len(self.wire_colors)):
                    if self.wire_colors[i] == "blue":
                        last_blue = i
                answer = last_blue
            else:
                answer = len(self.wire_colors) - 1
        elif self.n_wires == 4:
            if "red" in wire_counts and wire_counts["red"] > 1 and (self.serial_number % 10) % 2 == 1:
                answer = 0
                for i, color in enumerate(self.wire_colors): # Find last red wire
                    if color == "red":
                        answer = i
            elif self.wire_colors[-1] == "yellow" and "red" not in wire_counts:
                answer = 0
            elif "blue" in wire_counts and wire_counts["blue"] == 1:
                answer = 0
            elif "yellow" in wire_counts and wire_counts["yellow"] > 1:
                answer = 3
            else:
                answer = 1
        elif self.n_wires == 5:
            if self.wire_colors[-1] == "black" and (self.serial_number % 10) % 2 == 1:
                answer = 3
            elif "red" in wire_counts and wire_counts["red"] == 1 and "yellow" in wire_counts and wire_counts["yellow"] > 1:
                answer = 0
            elif "black" not in wire_counts:
                answer = 1
            else:
                answer = 0
        elif self.n_wires == 6:
            if "yellow" not in wire_counts and (self.serial_number % 10) % 2 == 1:
                answer = 2
            elif "yellow" in wire_counts and wire_counts["yellow"] == 1 and "white" in wire_counts and wire_counts["white"] > 1:
                answer = 3
            elif "red" not in wire_counts:
                answer = 5
            else:
                answer = 3
        
        if answer == self.chosen_index:
            return True
        return False
    
    def on_wire_cut(self, event=None, wire_num=0):
        self.chosen_index = wire_num
        self.feedback = self.finish_check()
   
    def __str__(self):
        return "SimpleWirePuzzle"
            
