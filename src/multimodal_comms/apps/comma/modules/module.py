import pygame
from multimodal_comms.apps.comma.paths import SOUND_ROOT
try:
    import pyautogui
except ModuleNotFoundError:
    pyautogui = None

class Module:
    """Base class for puzzles"""
    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.finished = False
        self.success = False
        self.solver_private_info = ""
        self.width = self.game_manager.width
        self.height = self.game_manager.height
        self.buttons = []
        self.mistakes = 0
        self.state = {}
        self.actions = []

        self.start_seconds = self.game_manager.total_seconds
        self.serial_number = game_manager.serial_number
        self.timer_label = game_manager.timer_label
        self.canvas = game_manager.canvas
        self.feedback = ""
        self.canvas.pack()

        if self.game_manager.use_sound:
            pygame.mixer.init()

    def execute_action(self, potential_action):
        """Input: The Solver's Message
        This function should parse the solver's message and invoke any actions the solver wants in the message.
        Returns True if the solver executed an action. Returns False otherwise."""
        for action in self.actions:
            cur_mistakes = self.mistakes

            if self.game_manager.action_delimeter + action.lower() in potential_action.lower():
                if self.finished:
                    return True
                self.actions[action]()
                
                post_mistakes = self.mistakes
                
                if post_mistakes > cur_mistakes:
                    message = "That action seems to have been a mistake."
                else:
                    message = "The action was performed successfully."

                self.game_manager.display_message(message, "environment")
                self.game_manager.add_to_conversations({"from": "ENVIRONMENT", "value": message}, role="solver")
                return True
            
        return False

            
    # Returns a bool indicating whether the puzzle is finished
    def check_correct(self):
        return False

    def update_feedback(self):
        self.feedback = self.finish_check()
    
    def click_button(self, x, y):
        if pyautogui is None:
            raise RuntimeError("automated GUI clicking requires the full Conda environment")
        pyautogui.moveTo(x, y, duration=0.1)
        pyautogui.click()

    # Draws a red circle on the module and plays a sound to indicate a mistake was made
    # Also increments the total mistakes on the module by 1
    def log_mistake(self):
        if self.game_manager.use_sound:
            sound = pygame.mixer.Sound(str(SOUND_ROOT / "wrong.mp3"))
            sound.set_volume(0.1)

            # Play the sound
            sound.play()

        self.mistakes += 1
        self.canvas.create_oval(
            self.width - 100, 100, self.width - 50, 50, outline="black", width=2, fill="red")

        self.canvas.after(1000, lambda: self.canvas.create_oval(
            self.width - 100, 100, self.width - 50, 50, outline="black", width=2, fill="gray"))
        
        self.feedback = "That action seems to have been a mistake."
        
    def finish_check(self):
        correct = self.check_correct()
        if correct:
            self.canvas.create_oval(
                self.width - 100, 100, self.width - 50, 50, outline="black", width=2, fill="green")
            # print(f"Made {self.mistakes} mistakes solving this puzzle")
            # print(
            #     f"Took {self.start_seconds - self.game_manager.total_seconds} seconds to solve this puzzle")

            if self.game_manager.use_sound:
                sound = pygame.mixer.Sound(str(SOUND_ROOT / "correct.mp3"))
                sound.play()

            for button in self.buttons:
                button.destroy()

            if hasattr(self, 'stage_labels'):
                for label in self.stage_labels:
                    label.destroy()
            
            self.finished = True
            self.success = True       
            return "The puzzle is solved! Here comes a new puzzle. Let's start working on it."
        else:
            self.log_mistake()

            return "Wrong action! Please try again."
        
    def get_score(self):
        return None
