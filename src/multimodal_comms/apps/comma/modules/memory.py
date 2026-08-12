import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
from multimodal_comms.apps.comma.modules.util import HighlightButton
import random
import functools

class MemoryPuzzle(Module):
    def __init__(self, game_manager):
        super(MemoryPuzzle, self).__init__(game_manager)

        self.actions = {}
        action_names = ["press_1", "press_2", "press_3", "press_4"]
        for action in action_names:
            self.actions[action] = functools.partial(self.on_button_press, None, int(action.split("_")[1]))
        
        self.stage = 1
        self.max_stage = 0

        display_width = 330
        display_height = 200
        display_top_left_x = 30
        display_top_left_y = 100

        if self.game_manager.config:
            self.display_text = str(self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["display_text"])
        else:
            self.display_text = str(random.randint(1, 4))
        self.history = []

        display = self.canvas.create_rectangle(display_top_left_x, display_top_left_y, display_top_left_x +
                                               display_width, display_top_left_y + display_height, outline=None, width=2, fill="black")
        self.display_text_object = self.canvas.create_text(
            display_top_left_x + display_width / 2, display_top_left_y + display_height / 2, text=self.display_text, fill="#33ff00", font=("Helvetica", 50))

        self.button_texts = ["3", "1", "2", "4"]
        for i in range(4):
            button_width = display_width / 5
            pad = 20
            button = HighlightButton(self, display_top_left_x + (button_width + pad) * i, 330, 75, 40, text=self.button_texts[i], font=(
                "Helvetica", 20), command=functools.partial(self.game_manager.solver.step, f"press_{self.button_texts[i]}"))
            self.buttons.append(button)

        self.canvas.create_rectangle(
            385, 150, 465, 384, outline='white', width=2, fill="black")

        self.lights = []
        for i in range(5):
            light = self.canvas.create_rectangle(
                395, 354 - i * 30, 455, 374 - i * 30, outline=None, width=2, fill="gray")
            self.lights.append(light)

    def on_button_press(self, event=None, pressed_button=1):
        correct_button = None

        if self.stage == 1:
            if self.display_text == "1":
                correct_button = 1
            elif self.display_text == "2":
                correct_button = 1
            elif self.display_text == "3":
                correct_button = 2
            elif self.display_text == "4":
                correct_button = 4

        elif self.stage == 2:
            if self.display_text == "1":
                correct_button = 4
            elif self.display_text == "2":
                correct_button = self.history[0]
            elif self.display_text == "3":
                correct_button = 3
            elif self.display_text == "4":
                correct_button = self.history[0]

        elif self.stage == 3:
            if self.display_text == "1":
                correct_button = self.history[1]
            elif self.display_text == "2":
                correct_button = self.history[0]
            elif self.display_text == "3":
                correct_button = 2
            elif self.display_text == "4":
                correct_button = 4

        elif self.stage == 4:
            if self.display_text == "1":
                correct_button = self.history[0]
            elif self.display_text == "2":
                correct_button = 3
            elif self.display_text == "3":
                correct_button = self.history[1]
            elif self.display_text == "4":
                correct_button = self.history[1]

        elif self.stage == 5:
            if self.display_text == "1":
                correct_button = self.history[0]
            elif self.display_text == "2":
                correct_button = self.history[1]
            elif self.display_text == "3":
                correct_button = self.history[3]
            elif self.display_text == "4":
                correct_button = self.history[2]

        if correct_button == pressed_button:
            self.canvas.itemconfig(self.lights[self.stage - 1], fill="green")
            self.stage += 1
            self.max_stage = self.stage - 1
            self.history.append(correct_button)
            self.display_text = str(random.randint(1, 4))
            self.canvas.itemconfig(
                self.display_text_object, text=self.display_text)
            if self.stage == 6:
                self.finish_check()

        else:
            self.log_mistake()
            self.stage = 1
            for i in range(len(self.lights)):
                self.canvas.itemconfig(self.lights[i], fill="gray")
            self.history = []
            self.display_text = str(random.randint(1, 4))
            self.canvas.itemconfig(
                self.display_text_object, text=self.display_text)

    def check_correct(self):
        return True

    def get_score(self):
        return self.max_stage / 5

    def __str__(self):
        return "MemoryPuzzle"
