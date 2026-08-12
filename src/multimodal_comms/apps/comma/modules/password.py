import tkinter as tk
from multimodal_comms.apps.comma.modules.module import Module
import random
import functools
from multimodal_comms.apps.comma.modules.util import HighlightButton


class PasswordPuzzle(Module):
    """he solver cycles through letters to form a valid word from a
    predefined list, submitting the correct word to complete the puzzle.
    """
    def __init__(self, game_manager):
        super(PasswordPuzzle, self).__init__(game_manager)

        self.actions = {}
        for i in range(5):
            self.actions[f"up_button_{i+1}"] = functools.partial(self.on_triangle_click, None, i, True)
            self.actions[f"down_button_{i+1}"] = functools.partial(self.on_triangle_click, None, i, False)
        self.actions["submit"] = self.on_submit


        possible_words = ["about", "after", "again", "below", "could", "every", "first", "found", "great", "house", "large", "learn", "never", "other", "place", "plant", "point",
                          "right", "small", "sound", "spell", "still", "study", "their", "there", "these", "thing", "think", "three", "water", "where", "which", "world", "would", "write"]
        
        if self.game_manager.config:
            self.answer_word = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["answer_word"]
        else:
            self.answer_word = random.choice(possible_words)
        
        self.letter_possibilities = []
        for letter in self.answer_word:
            ascii_val = ord(letter)

            # Calculate the previous and next characters with wrap-around
            prev_char = str(chr((ascii_val - ord('a') - 1) % 26 + ord('a')))
            next_char = str(chr((ascii_val - ord('a') + 1) % 26 + ord('a')))
            self.letter_possibilities.append([prev_char, letter, next_char])


        if self.game_manager.config:
            self.current_letters = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["current_letters"]
        else:
            self.current_letters = [random.choice(x) for x in self.letter_possibilities]
        

        self.up_buttons = []
        self.down_buttons = []
        self.buttons = []

        # Upward buttons
        start_x, start_y = 70, 150
        w = (self.width - start_x) // 5
        self.t_width = 15
        for i in range(5):
            triangle = self.canvas.create_polygon(start_x, start_y - self.t_width, start_x + self.t_width, start_y +
                                                  self.t_width, start_x - self.t_width, start_y + self.t_width, fill='black', outline='black')
            self.canvas.tag_bind(triangle, '<Button-1>', functools.partial(
                self.game_manager.solver.step, f"up_button_{i+1}"))
            start_x += w
            self.up_buttons.append(triangle)

        # Boxes around letters
        start_x, start_y = 70, 175
        for _ in range(5):
            box = self.canvas.create_rectangle(
                start_x - 43, start_y, start_x + 43, 375, outline="black", width=2, fill="white")
            start_x += w

        # Drawing letters
        start_x, start_y = 70, 275
        self.letters = []

        for i in range(5):
            letter = self.canvas.create_text(
                start_x, start_y, text=self.current_letters[i].upper(), font=("Arial", 50))

            self.letters.append(letter)
            start_x += w

        # Downwards facing buttons
        start_x, start_y = 70, 400
        for i in range(5):
            triangle = self.canvas.create_polygon(start_x, start_y + self.t_width, start_x + self.t_width, start_y -
                                                  self.t_width, start_x - self.t_width, start_y - self.t_width, fill='black', outline='black')
            self.canvas.tag_bind(triangle, '<Button-1>', functools.partial(
                self.game_manager.solver.step, f"down_button_{i+1}"))
            start_x += w
            self.down_buttons.append(triangle)

        # Submit Button
        send_button = HighlightButton(
            self, self.width / 2 - 80, 425, 160, 40, text="SUBMIT", command=functools.partial(
                self.game_manager.solver.step, f"submit"))
        self.buttons.append(send_button)

    def on_triangle_click(self, event=None, letter_idx=0, increment=True):
        letter = self.canvas.itemcget(self.letters[letter_idx], "text").lower()
        inc = 1 if increment else -1
        idx = (self.letter_possibilities[letter_idx].index(
            letter) + inc) % len(self.letter_possibilities[letter_idx])

        self.canvas.delete(self.letters[letter_idx])
        start_x, start_y = 70, 275
        w = (self.width - start_x) // 5
        self.letters[letter_idx] = self.canvas.create_text(
            start_x + w * letter_idx, start_y, text=self.letter_possibilities[letter_idx][idx].upper(), font=("Arial", 50))

    def on_submit(self, event=None):
        if self.finish_check().startswith("The puzzle is solved"):
            for letter in self.letters:
                self.canvas.delete(letter)

    def check_correct(self):
        for i, letter in enumerate(self.letters):
            character = self.canvas.itemcget(letter, "text").lower()
            if character != self.answer_word[i]:
                return False
        return True

    def __str__(self):
        return "PasswordPuzzle"
