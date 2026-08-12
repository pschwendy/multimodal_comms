import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
import random
import functools
from multimodal_comms.apps.comma.modules.util import HighlightButton
import pyautogui

class WhoPuzzle(Module):
    """The solver must tell the value on a display to the expert, who will identify
    a button label. The solver must then tell this label to the expert, and then press the correct button
    based on a detailed list of instructions.
    """
    def __init__(self, game_manager):
        super(WhoPuzzle, self).__init__(game_manager)

        self.actions = {}
        action_names = ["press_top_left_button", "press_top_right_button", "press_middle_left_button",  "press_middle_right_button", "press_bottom_left_button", "press_bottom_right_button"]
        for i, action in enumerate(action_names):
            self.actions[action] = functools.partial(self.on_image_click, None, i)
        
        self.display_mapping = {"YES":2, "FIRST":1, "DISPLAY":5, "OKAY":1, "SAYS":5, "NOTHING":2, "":4, "BLANK":3, "NO":5, "LED":2, "LEAD":5, "READ":3, "RED":3, "REED":4, "LEED":4, "HOLD ON":5, "YOU":3, "YOU ARE":5, "YOUR":3, "YOU'RE": 3, "UR":0, "THERE":5, "THEY'RE":4, "THEIR":3, "THEY ARE":2, "SEE":5, "C":1, "CEE":5}
        if self.game_manager.config:
            self.display_text = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["display_text"]
        else:
            self.display_text = random.choice(list(self.display_mapping.keys()))
        

        self.canvas.create_rectangle(30, 30, self.width - 120, 150, outline="black", width=2, fill="black")
        self.canvas.create_text((self.width - 90) // 2, 90, text=self.display_text, fill="#33ff00", font=("Helvetica", 30))
        
        self.word_to_list = {"READY": ["YES", "OKAY", "WHAT", "MIDDLE", "LEFT", "PRESS", "RIGHT", "BLANK", "READY", "NO", "FIRST", "UHHH", "NOTHING", "WAIT"], 
        "FIRST": ["LEFT", "OKAY", "YES", "MIDDLE", "NO", "RIGHT", "NOTHING", "UHHH", "WAIT", "READY", "BLANK", "WHAT", "PRESS", "FIRST"],
        "NO": ["BLANK", "UHHH", "WAIT", "FIRST", "WHAT", "READY", "RIGHT", "YES", "NOTHING", "LEFT", "PRESS", "OKAY", "NO", "MIDDLE"],
        "BLANK": ["WAIT", "RIGHT", "OKAY", "MIDDLE", "BLANK", "PRESS", "READY", "NOTHING", "NO", "WHAT", "LEFT", "UHHH", "YES", "FIRST"],
        "NOTHING": ["UHHH", "RIGHT", "OKAY", "MIDDLE", "YES", "BLANK", "NO", "PRESS", "LEFT", "WHAT", "WAIT", "FIRST", "NOTHING", "READY"],
        "YES": ["OKAY", "RIGHT", "UHHH", "MIDDLE", "FIRST", "WHAT", "PRESS", "READY", "NOTHING", "YES", "LEFT", "BLANK", "NO", "WAIT"],
        "WHAT": ["UHHH", "WHAT", "LEFT", "NOTHING", "READY", "BLANK", "MIDDLE", "NO", "OKAY", "FIRST", "WAIT", "YES", "PRESS", "RIGHT"],
        "UHHH": ["READY", "NOTHING", "LEFT", "WHAT", "OKAY", "YES", "RIGHT", "NO", "PRESS", "BLANK", "UHHH", "MIDDLE", "WAIT", "FIRST"],
        "LEFT": ["RIGHT", "LEFT", "FIRST", "NO", "MIDDLE", "YES", "BLANK", "WHAT", "UHHH", "WAIT", "PRESS", "READY", "OKAY", "NOTHING"],""
        "RIGHT": ["YES", "NOTHING", "READY", "PRESS", "NO", "WAIT", "WHAT", "RIGHT", "MIDDLE", "LEFT", "UHHH", "BLANK", "OKAY", "FIRST"],
        "MIDDLE": ["BLANK", "READY", "OKAY", "WHAT", "NOTHING", "PRESS", "NO", "WAIT", "LEFT", "MIDDLE", "RIGHT", "FIRST", "UHHH", "YES"],
        "OKAY": ["MIDDLE", "NO", "FIRST", "YES", "UHHH", "NOTHING", "WAIT", "OKAY", "LEFT", "READY", "BLANK", "PRESS", "WHAT", "RIGHT"],
        "WAIT": ["UHHH", "NO", "BLANK", "OKAY", "YES", "LEFT", "FIRST", "PRESS", "WHAT", "WAIT", "NOTHING", "READY", "RIGHT", "MIDDLE"],
        "PRESS": ["RIGHT", "MIDDLE", "YES", "READY", "PRESS", "OKAY", "NOTHING", "UHHH", "BLANK", "LEFT", "FIRST", "WHAT", "NO", "WAIT"],
        "YOU": ["SURE", "YOU ARE", "YOUR", "YOU'RE", "NEXT", "UH HUH", "UR", "HOLD", "WHAT?", "YOU", "UH UH", "LIKE", "DONE", "U"],
        "YOU ARE": ["YOUR", "NEXT", "LIKE", "UH HUH", "WHAT?", "DONE", "UH UH", "HOLD", "YOU", "U", "YOU'RE", "SURE", "UR", "YOU ARE"],
        "YOUR": ["UH UH", "YOU ARE", "UH HUH", "YOUR", "NEXT", "UR", "SURE", "U", "YOU'RE", "YOU", "WHAT?", "HOLD", "LIKE", "DONE"],
        "YOU'RE": ["YOU", "YOU'RE", "UR", "NEXT", "UH UH", "YOU ARE", "U", "YOUR", "WHAT?", "UH HUH", "SURE", "DONE", "LIKE", "HOLD"],
        "UR": ["DONE", "U", "UR", "UH HUH", "WHAT?", "SURE", "YOUR", "HOLD", "YOU'RE", "LIKE", "NEXT", "UH UH", "YOU ARE", "YOU"],
        "U": ["UH HUH", "SURE", "NEXT", "WHAT?", "YOU'RE", "UR", "UH UH", "DONE", "U", "YOU", "LIKE", "HOLD", "YOU ARE", "YOUR"],
        "UH HUH": ["UH HUH", "YOUR", "YOU ARE", "YOU", "DONE", "HOLD", "UH UH", "NEXT", "SURE", "LIKE", "YOU'RE", "UR", "U", "WHAT?"],
        "UH UH": ["UR", "U", "YOU ARE", "YOU'RE", "NEXT", "UH UH", "DONE", "YOU", "UH HUH", "LIKE", "YOUR", "SURE", "HOLD", "WHAT?"],
        "WHAT?": ["YOU", "HOLD", "YOU'RE", "YOUR", "U", "DONE", "UH UH", "LIKE", "YOU ARE", "UH HUH", "UR", "NEXT", "WHAT?", "SURE"],
        "DONE": ["SURE", "UH HUH", "NEXT", "WHAT?", "YOUR", "UR", "YOU'RE", "HOLD", "LIKE", "YOU", "U", "YOU ARE", "UH UH", "DONE"],
        "NEXT": ["WHAT?", "UH HUH", "UH UH", "YOUR", "HOLD", "SURE", "NEXT", "LIKE", "DONE", "YOU ARE", "UR", "YOU'RE", "U", "YOU"],
        "HOLD": ["YOU ARE", "U", "DONE", "UH UH", "YOU", "UR", "SURE", "WHAT?", "YOU'RE", "NEXT", "HOLD", "UH HUH", "YOUR", "LIKE"],
        "SURE": ["YOU ARE", "DONE", "LIKE", "YOU'RE", "YOU", "HOLD", "UH HUH", "UR", "SURE", "U", "WHAT?", "NEXT", "YOUR", "UH UH"],
        "LIKE": ["YOU'RE", "NEXT", "U", "UR", "HOLD", "DONE", "UH UH", "WHAT?", "UH HUH", "YOU", "LIKE", "SURE", "YOU ARE", "YOUR"]}
        
        # Create a list to hold button objects
        self.buttons = []

        self.button_width = 160
        self.button_height = 80

        if self.game_manager.config:
            self.button_texts = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["button_texts"]
        else:
            self.button_texts = random.sample(list(self.word_to_list.keys()), 6)
        button_texts_idx = self.display_mapping[self.display_text]

        word_list = self.word_to_list[self.button_texts[button_texts_idx]]

        for word in word_list:
            if word in self.button_texts:
                self.answer = self.button_texts.index(word)
                break
            
        for i in range(6):
            button = HighlightButton(self, 30 + 190 * (i % 2), 170 + 90 * (i // 2), self.button_width, self.button_height, text=self.button_texts[i], command=functools.partial(self.game_manager.solver.step, action_names[i]))
            self.buttons.append(button)
        
    def on_image_click(self, event=None, img_num = 0):
        self.prediction = img_num 
        self.finish_check()

    def check_correct(self):
        return self.answer == self.prediction

    def __str__(self):
        return "WhoPuzzle"
