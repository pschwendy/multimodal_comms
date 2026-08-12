from multimodal_comms.apps.comma.modules.module import Module
import os
import random
import functools
from multimodal_comms.apps.comma.modules.util import HighlightButton
from PIL import Image, ImageTk

class DogPuzzle(Module):
    """Puzzle where a picture containing 0-4 dogs will be displayed. The solver needs to wait and press
    the submit button when the last digit of the timer matches the number of dogs in the image.
    """
    def __init__(self, game_manager):
        super(DogPuzzle, self).__init__(game_manager)
        
        self.actions = {}
        for i in range(10):
            self.actions[f"press_button_after_{i}_seconds"] = functools.partial(self.press_button_after, i)
            
        dog_images = os.listdir(os.path.join("images", "dogs"))
        if self.game_manager.config:
            sampled_image = dog_images[self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["dog_image_num"]]
        else:
            sampled_image = random.choice(dog_images)

        pil_image = Image.open(os.path.join("images", "dogs", sampled_image))
        pil_image = pil_image.resize((450,300))
        tk_image = ImageTk.PhotoImage(pil_image)
        
        self.image_id = self.canvas.create_image(250, 260, image = tk_image, anchor='center')
        self.canvas.image = tk_image
        self.n_dogs = int(sampled_image.split("_")[0])

        send_button = HighlightButton(
            self, self.width / 2 - 80, 425, 160, 40, text="SUBMIT", command=functools.partial(self.game_manager.solver.step, "press_button_after_0_seconds"))
        self.buttons.append(send_button)

    def on_submit(self, event=None):
        if self.finish_check().startswith("The puzzle is solved!"):
            self.canvas.delete(self.image_id)

    def press_button_after(self, seconds):
        self.game_manager.total_seconds -= seconds
        minutes, seconds = divmod(self.game_manager.total_seconds, 60)
        self.game_manager.timer_label.config(text=f"Time Left: {minutes:02d}:{seconds:02d}")
        self.on_submit()

    def check_correct(self):
        timer_text = self.timer_label.cget("text")
        correct = False
        if self.n_dogs == 4 and "4" in timer_text:
            correct = True
        if self.n_dogs == 3 and "3" in timer_text:
            correct = True
        if self.n_dogs == 2 and "2" in timer_text:
            correct = True
        if self.n_dogs == 1 and "1" in timer_text:
            correct = True
        if self.n_dogs == 0 and "0" in timer_text:
            correct = True
        return correct


    def __str__(self):
        return "DogPuzzle"
