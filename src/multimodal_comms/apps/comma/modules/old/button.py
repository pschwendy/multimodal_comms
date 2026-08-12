import math
import random
import functools
from multimodal_comms.apps.comma.modules.module import Module
import pyautogui
import time


class ButtonPuzzle(Module):
    def __init__(self, game_manager):
        super(ButtonPuzzle, self).__init__(game_manager)

        button_colors = ["yellow"]
        strip_colors = ["red", "white", "yellow", "blue", "green"]
        button_texts = ["PRESS"]

        if self.game_manager.config:
            config = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]
            self.strip_color = config["strip_color"]
            self.button_color = config["button_color"]
            self.button_text = config["button_text"]
            
        else:
            self.strip_color = random.choice(strip_colors)
            self.button_color = random.choice(button_colors)
            self.button_text = random.choice(button_texts)
            

        self.clicked_button = False

        self.canvas.create_rectangle(
            25, 125, self.width - 150, self.height - 50, outline="black", width=2)
        self.canvas.create_rectangle(
            55, 155, self.width - 180, self.height - 80, outline="black", width=2)
        self.canvas.create_line(self.width - 150, 125,
                                self.width - 180, 155, width=2)
        self.canvas.create_line(25, 125, 55, 155, width=2)

        # Create the switch
        self.switch_on_y = 155  
        self.switch_off_y = 375  
        self.switch_x1 = 55
        self.switch_x2 = self.width - 180
        self.switch_height = 50

        self.switch_rect = self.canvas.create_rectangle(
            self.switch_x1, self.switch_off_y, self.switch_x2, self.switch_off_y + self.switch_height,
            outline="black", width=2, fill=self.button_color)

        self.switch_text = self.canvas.create_text(
            self.switch_x1 + 132, self.switch_off_y + 25, text="OFF", fill="black", font=("Helvetica", 20))

        self.canvas.create_rectangle(
            self.width - 100, 200, self.width - 50, self.height - 50, outline="black", width=2)

        # Left mouse button press
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        # Left mouse button release
        # self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def send_info_to_solver(self):
        self.actions.append({"type": "button", "name": f"turn_on_switch", "x_coord": (
                55 + 320) / 2, "y_coord": (155 + 420) / 2})
        self.actions.append({"type": "button", "name": f"turn_off_switch", "x_coord": (
                55 + 320) / 2, "y_coord": (155 + 420) / 2})
        # for i in range(10):
        #     self.actions.append({"type": "press_and_hold", "name": f"press_and_hold_button_{i}_seconds", "x_coord": (
        #         55 + 320) / 2, "y_coord": (155 + 420) / 2})
        super().send_info_to_solver()

    def execute_action(self, message):
        if message == "turn_off_switch":
            self.click_button(self.switch_x1 + 10, self.switch_on_y + 10)
            return True
        if message == "turn_on_switch":
            self.click_button(self.switch_x1 + 10, self.switch_off_y + 10)
            return True
        # for i in range(10):
        #     if f"press_and_hold_button_{i}_seconds" in message:
        #         self.click_button(((55 + 320) / 2), ((155 + 420) / 2), i)
        #         return True

        return False

    def on_press(self, event):
        x1, x2 = self.switch_x1, self.switch_x2
        y1, y2 = 0, 0
        if not self.clicked_button:
            y1, y2 = self.switch_off_y, self.switch_off_y + self.switch_height
        else:
            y1, y2 = self.switch_on_y, self.switch_on_y + self.switch_height

        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            if not self.clicked_button:
                self.canvas.create_rectangle(self.width - 100, 200, self.width - 50,
                                            self.height - 50, outline="black", width=2, fill=self.strip_color)
                self.canvas.move(self.switch_rect, 0, -(self.switch_off_y - self.switch_on_y))
                self.canvas.itemconfig(self.switch_rect, fill=self.button_color)  
                self.canvas.itemconfig(self.switch_text, text="ON")  
                self.canvas.move(self.switch_text, 0, -(self.switch_off_y - self.switch_on_y))
                self.clicked_button = True
            else:
                self.canvas.create_rectangle(
                    self.width - 100, 200, self.width - 50, self.height - 50, outline="black", width=2, fill="gray")
                self.canvas.move(self.switch_rect, 0, (self.switch_off_y - self.switch_on_y))
                self.canvas.itemconfig(self.switch_rect, fill=self.button_color) 
                self.canvas.itemconfig(self.switch_text, text="OFF") 
                self.canvas.move(self.switch_text, 0, (self.switch_off_y - self.switch_on_y))
                self.clicked_button = False
                self.finish_check()

        # center_x = (55 + 320) / 2
        # center_y = (155 + 420) / 2

        # if not self.clicked_button:
        #     if math.sqrt(abs(event.x - center_x) ** 2 + abs(event.y - center_y) ** 2) < 132:
        #         self.canvas.create_rectangle(self.width - 100, 200, self.width - 50,
        #                                     self.height - 50, outline="black", width=2, fill=self.strip_color)
        #         self.clicked_button = True

        # else:
        #     if math.sqrt(abs(event.x - center_x) ** 2 + abs(event.y - center_y) ** 2) < 132:
        #         self.clicked_button = False
        #         self.canvas.create_rectangle(
        #             self.width - 100, 200, self.width - 50, self.height - 50, outline="black", width=2, fill="gray")
        #         self.finish_check()

    def on_release(self, event):
        if self.clicked_button:
            self.clicked_button = False
            self.canvas.create_rectangle(
                self.width - 100, 200, self.width - 50, self.height - 50, outline="black", width=2, fill="gray")
            self.finish_check()

    def check_correct(self):
        timer_text = self.timer_label.cget("text")
        correct = False
        if self.button_text == "PRESS":
            if self.strip_color == "blue" and "4" in timer_text:
                correct = True
            if self.strip_color == "white" and "1" in timer_text:
                correct = True
            if self.strip_color == "yellow" and "5" in timer_text:
                correct = True
            if self.strip_color not in ["blue", "white", "yellow"] and "1" in timer_text:
                correct = True
        return correct

    def __str__(self):
        return "ButtonPuzzle"
