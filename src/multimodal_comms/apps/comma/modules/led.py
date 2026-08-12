import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
import random
import functools
from multimodal_comms.apps.comma.modules.util import HighlightButton

class LedPuzzle(Module):
    """The solver presses a button if the value of its letter, when multiplied
    by a stage's LED color multiplier and taken modulo 26, matches the value of the letter diagonally
    opposite it. At each stage, the letters on the buttons change.
    """
    def __init__(self, game_manager):
        super(LedPuzzle, self).__init__(game_manager)

        self.actions = {}
        self.actions['press_top_left_button'] = functools.partial(self.on_button_click, None, 0, 0)
        self.actions['press_top_right_button'] = functools.partial(self.on_button_click, None, 0, 1)
        self.actions['press_bottom_right_button'] = functools.partial(self.on_button_click, None, 1, 1)
        self.actions['press_bottom_left_button'] = functools.partial(self.on_button_click, None, 1, 0)
   
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.multiplier_table = {
            "red": 2,
            "green": 3,
            "blue": 4,
            "yellow": 5,
            "purple": 6,
            "orange": 7
        }

        
        self.current_stage = 0
        if self.game_manager.config:
            self.num_stages = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["num_stages"]
            self.led_colors = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["led_colors"]
        else:
            self.num_stages = random.randint(2, 5)
            
            self.led_colors = [random.choice(list(self.multiplier_table.keys())) for _ in range(self.num_stages)]
        
        self.stage_labels = []

        self.buttons = []

        self.create_widgets()

    def create_widgets(self):
        # Display the current stage number
        self.stage_label = tk.Label(
            self.canvas, text=f"Stage: {self.current_stage + 1}", font=("Arial", 10))
        self.stage_label.place(x=20, y=20)
        self.stage_labels.append(self.stage_label)

        # Display LEDs representing stages
        start_x = 100
        for i in range(self.num_stages):
            led = tk.Label(
                self.canvas, bg=self.led_colors[i], width=6, height=1)
            led.place(x=start_x, y=20)
            self.stage_labels.append(led)
            start_x += 75

        self.setup_buttons()

    def setup_buttons(self):
        self.buttons = []
        start_x, start_y = 120, 120
        names = ['press_top_left_button', 'press_top_right_button', 'press_bottom_left_button', 'press_bottom_right_button']
        for i in range(2):
            for j in range(2):
                button = HighlightButton(self, start_x + j * 140, start_y + i * 140, 120,
                                         120, text="", command=functools.partial(self.game_manager.solver.step, names[i * 2 + j]))
                self.buttons.append(button)
        self.set_correct_letters()

    def set_correct_letters(self):
        correct_i, correct_j = random.randint(0, 1), random.randint(0, 1)
        multiplier = self.multiplier_table[self.led_colors[self.current_stage]]

        correct_letter_value = random.randint(0, 25)
        correct_letter = self.letters[correct_letter_value]
        self.buttons[correct_i * 2 + correct_j].config(text=correct_letter)

        diagonal_i, diagonal_j = 1 - correct_i, 1 - correct_j
        diagonal_value = (correct_letter_value * multiplier) % 26
        diagonal_letter = self.letters[diagonal_value]
        self.buttons[diagonal_i * 2 + diagonal_j].config(text=diagonal_letter)

        remaining_buttons = [0, 1, 2, 3]
        remaining_buttons.remove(correct_i * 2 + correct_j)
        remaining_buttons.remove(diagonal_i * 2 + diagonal_j)
        for idx in remaining_buttons:
            self.buttons[idx].config(text=random.choice(self.letters))

    def calculate_correct_letter(self, i, j):
        current_letter = self.buttons[i * 2 + j].cget("text")
        letter_value = self.letters.index(current_letter)
        multiplier = self.multiplier_table[self.led_colors[self.current_stage]]
        correct_value = (letter_value * multiplier) % 26
        diagonal_i, diagonal_j = (1 - i, 1 - j)
        diagonal_letter = self.buttons[diagonal_i *
                                       2 + diagonal_j].cget("text")
        diagonal_value = self.letters.index(diagonal_letter)
        return correct_value == diagonal_value

    def check_correct(self):
        return True

    def on_button_click(self, event=None, i=0, j=0):
        if self.calculate_correct_letter(i, j):
            self.current_stage += 1
            if self.current_stage >= self.num_stages:
                self.stage_label.destroy()
                for led in self.stage_labels:
                    led.destroy()
                self.finish_check()
            else:
                self.stage_label.config(
                    text=f"Stage: {self.current_stage + 1}")
                self.set_correct_letters()
        else:
            self.log_mistake()

    def get_button_coordinates(self, button):
        return button.winfo_rootx() + 10, button.winfo_rooty() + 10

    def get_score(self):
        return self.current_stage / self.num_stages

    def __str__(self):
        return "LedPuzzle"
