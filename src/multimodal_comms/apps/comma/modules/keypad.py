import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
import os
from multimodal_comms.apps.comma.paths import IMAGE_ROOT
import random
import functools
from multimodal_comms.apps.comma.modules.util import HighlightButton


class KeyPadPuzzle(Module):
    """The solver must describe the symbol of each button in a 2x2 grid. The expert must then identify a column in the manual containing these four unique symbols and tell the solver to press the symbols in the correct order from top to bottom.
    """
    def __init__(self, game_manager):
        super(KeyPadPuzzle, self).__init__(game_manager)

        self.actions = {}
        action_names = ["press_top_left_button", "press_bottom_left_button", "press_top_right_button", "press_bottom_right_button"]
        for i, action in enumerate(action_names):
            self.actions[action] = functools.partial(self.on_image_click, None, i)
        
        image_collections = []
        keypad_root = IMAGE_ROOT / "keypad"
        for folder in os.listdir(keypad_root):
            image_strip = []
            for file in os.listdir(keypad_root / folder):
                image_path = str(keypad_root / folder / file)
                image_strip.append(image_path)

            image_collections.append(image_strip)


        if self.game_manager.config:
            strip = image_collections[self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["strip_num"]]
            image_indices = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["image_nums"]
            images = [strip[x] for x in image_indices]
        else:
            strip = random.choice(image_collections)
            images = random.sample(strip, 4)
            random.shuffle(images)

        correct_order = sorted(images)
        self.correct_buttons = [images.index(x) for x in correct_order]
        self.current_index = 0

        self.buttons = []
        for i in range(len(images)):
            image = tk.PhotoImage(file=images[i])
            resized = image.subsample(3)
            button = HighlightButton(self, 40 + 180 * (i // 2), 100 + 180 * (i % 2), 166,
                                     166, image=resized, command=functools.partial(self.game_manager.solver.step, action_names[i]))
            button.image = resized
            self.buttons.append(button)

        self.max_score = 0

    def on_image_click(self, event=None, img_num=0):
        if self.current_index == len(self.correct_buttons):
            self.current_index = 0
            self.score = 0
            return

        if self.correct_buttons[self.current_index] == img_num:
            self.current_index += 1
            self.max_score = self.current_index
        else:
            self.current_index = 0
            self.score = 0
            self.log_mistake()

        if self.current_index == len(self.correct_buttons):
            self.finish_check()

    def check_correct(self):
        return True

    def get_score(self):
        return self.max_score / 4

    def __str__(self):
        return "KeyPadPuzzle"
