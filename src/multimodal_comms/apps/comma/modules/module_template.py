from multimodal_comms.apps.comma.modules.module import Module
from multimodal_comms.apps.comma.modules.util import HighlightButton
import functools

class ExamplePuzzle(Module):
    """Template for adding a puzzle to our framework.Remember to also add the
instructions for the puzzle in natural text to config/prompts.json.
Optionally, if the instructions require an image, you may include an image of
the puzzle in images/manuals/<Name of your puzzle>.jpg. The name should be the
same as the string representation of this class (see bottom)."""
    def __init__(self, game_manager):
        super(ExamplePuzzle, self).__init__(game_manager)

        # Define action dictionary (the keys will be presented to the solver)
        # Map each action to its corresponding handler with functools.partial
        self.actions = {"press_button": functools.partial(self.on_button_press)}

        if self.game_manager.config: # If you have some config for the puzzle, you can load it here
            my_config = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]
        else: # Some initialization logic
            pass

        # Create some buttons / canvas elements if you like
        # This is mainly for human settings if the human wants to click on the GUI
        # This defines a button, which has a handler called on_submit()
        send_button = HighlightButton(
            self, 
            self.width / 2 - 80,    # Top Left X
            425,                    # Top Left Y
            160,                    # Button Width
            40,                     # Button Height
            text="SUBMIT",          # Text Displayed on Button
            command=functools.partial(self.game_manager.solver.step, "press_button")) # Handler after pressing button
        
        # When adding buttons, make sure to add them to self.buttons
        self.buttons.append(send_button)

    def on_button_press(self):
        """Example handler for tkinter event"""
        self.finish_check() # This function checks if the puzzle in its current state if correct and finished.

    def check_correct(self):
        """Function to check if the current state of the puzzle is the final correct one.
        Called when finish_check() is called."""
        return True
    
    def __str__(self):
        """String representation of the puzzle"""
        return "ExamplePuzzle"
