import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
import random
import pygame
import pyautogui


class SoundPuzzle(Module):
    def __init__(self, game_manager):
        super(SoundPuzzle, self).__init__(game_manager)
        self.sound_clips = {
            "Taxi Dispatch": "&&&**",
            "Cow": "&$#$&",
            "Extractor Fan": "$#$*&",
            "Train Station": "#$$**"
        }
        # random.choice(list(self.sound_clips.keys()))
        self.current_clip = "Cow"
        self.entered_code = []
        self.buttons = []
        self.submit_button = None
        self.play_button = None
        self.x_offset = 150
        self.y_offset = 150
        self.create_widgets()
        pygame.mixer.init()

    def create_widgets(self):
        # Create the secondary canvas for the keypad
        self.keypad_canvas = tk.Canvas(
            self.canvas, width=self.width, height=self.height)
        self.canvas.create_window(
            (self.x_offset, self.y_offset), window=self.keypad_canvas, anchor="nw")

        # Create the play button
        self.play_button = tk.Button(
            self.keypad_canvas, text="Play", command=self.play_sound)
        self.play_button.grid(row=0, column=1, pady=10)

        # Create keypad buttons
        symbols = ["#", "$", "&", "*"]
        for i, symbol in enumerate(symbols):
            button = tk.Button(self.keypad_canvas, text=symbol,
                               command=lambda s=symbol: self.add_to_code(s))
            button.grid(row=1, column=i, padx=5, pady=5)
            self.buttons.append(button)

        # Create the submit button
        self.submit_button = tk.Button(
            self.keypad_canvas, text="Submit", command=self.check_code)
        self.submit_button.grid(row=2, column=1, columnspan=2, pady=10)

        # Create a label to display the entered code
        self.code_display = tk.Label(self.keypad_canvas, text="Entered Code: ")
        self.code_display.grid(row=3, column=0, columnspan=4, pady=10)

    def play_sound(self):
        # Ensure your sound files are named accordingly
        sound_file = f"sounds/{self.current_clip}.mp3"
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        return sound_file

    def add_to_code(self, symbol):
        if len(self.entered_code) < 6:
            self.entered_code.append(symbol)
            self.update_code_display()

    def update_code_display(self):
        self.code_display.config(
            text="Entered Code: " + ''.join(self.entered_code))

    def check_code(self, headless_code=""):
        if ''.join(self.entered_code) == self.sound_clips[self.current_clip] or headless_code == self.sound_clips[self.current_clip]:
            self.finish_check()
        else:
            self.log_mistake()
            self.reset_code()

    def reset_code(self):
        self.entered_code = []
        self.update_code_display()

    def check_correct(self):
        return True

    def get_button_coordinates(self, button):
        canvas_x = self.keypad_canvas.winfo_x()
        canvas_y = self.keypad_canvas.winfo_y()
        button_x = button.winfo_x()
        button_y = button.winfo_y()
        return (canvas_x + button_x + 30, canvas_y + button_y + 80)

    def click_button(self, x, y):
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()

    def send_info_to_solver(self):
        xcord, ycord = self.get_button_coordinates(self.submit_button)
        self.actions.append({
            "type": "button",
            "name": "submit",
            "signature": f"check_code(headless_code: str) -> None",
            "x_coord": xcord,
            "y_coord": ycord
        })
        xcord, ycord = self.get_button_coordinates(self.play_button)
        self.actions.append({
            "type": "button",
            "name": "play_sound",
            "signature": f"play_sound() -> str",
            "x_coord": xcord,
            "y_coord": ycord
        })
        symbols = ["#", "$", "&", "*"]
        for i, symbol in enumerate(symbols):
            xcord, ycord = self.get_button_coordinates(self.buttons[i])
            self.actions.append({
                "type": "button",
                "name": f"enter_symbol_{symbol}",
                "signature": f"add_to_code(symbol: str) -> None",
                "x_coord": self.buttons[i].winfo_x() + self.x_offset,
                "y_coord": self.buttons[i].winfo_y() + self.y_offset
            })
        super().send_info_to_solver()

    def execute_action(self, message):
        if self.gui:
            for action in self.actions:
                if action["name"] in message:
                    self.click_button(x=action['x_coord'], y=action['y_coord'])
                    return True
            return False
        else:
            if message == "submit":
                self.check_code()
                return True
            if message == "play_sound":
                self.play_sound()
                return True
            symbols = ["#", "$", "&", "*"]
            for symbol in symbols:
                if message == f"enter_symbol_{symbol}":
                    self.add_to_code(symbol)
                    return True

            return False

        super.send_info_to_solver()

    def __str__(self):
        return "SoundPuzzle"
