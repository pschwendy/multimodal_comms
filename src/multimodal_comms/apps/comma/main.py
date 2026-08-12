import os
import tkinter as tk
from tkinter import font
import random
import json
import gc
import argparse
from sys import platform
from tqdm import tqdm
from multimodal_comms.apps.comma.modules import *
from multimodal_comms.apps.comma.agents import *

class GameManager:

    def __init__(self, master, args):

        # Defaul values for some hyperparameters
        self.MAX_CONVERSATION_TURNS = 20
        self.MAX_MISTAKES = 3
        self.EXPERT_MESSAGE_WORD_LIMIT = 100
        self.serial_number = random.randint(100000, 999999)
        self.total_seconds = 6000
        self.action_delimeter = ""
        self.starting_puzzle = 0
        self.ending_puzzle = None
        
        
        self.cur_step = 0
        self.use_sound = args.use_sound
        self.save_folder = args.save_folder
        self.master = master
        self.master.title("Puzzle")
        
        self.model_config = None
        if args.model_config:
            self.model_config = json.load(open(args.model_config))
            if "Hyperparameters" in self.model_config:
                self.serial_number = self.model_config["Hyperparameters"].get("SERIAL_NUMBER", self.serial_number)
                self.MAX_CONVERSATION_TURNS = self.model_config["Hyperparameters"].get("MAX_CONVERSATION_TURNS", self.MAX_CONVERSATION_TURNS)
                self.MAX_MISTAKES = self.model_config["Hyperparameters"].get("MAX_MISTAKES", self.MAX_MISTAKES)                
                self.total_seconds = self.model_config["Hyperparameters"].get("TOTAL_TIME", self.total_seconds)
                self.starting_puzzle = self.model_config["Hyperparameters"].get("STARTING_PUZZLE", 0)
                self.ending_puzzle = self.model_config["Hyperparameters"].get("END_PUZZLE", None)
                
                self.EXPERT_MESSAGE_WORD_LIMIT = self.model_config["Hyperparameters"].get("EXPERT_MESSAGE_WORD_LIMIT", self.EXPERT_MESSAGE_WORD_LIMIT)

        self.config = None
        if args.puzzle_config:
            self.config = json.load(open(args.puzzle_config))
            if self.ending_puzzle:
                self.config = self.config[self.starting_puzzle:self.ending_puzzle]
            else:
                self.config = self.config[self.starting_puzzle:]
            
        # Prevent resizing of the window
        self.master.resizable(False, False)
        self.master.geometry("+0+0")
        
        # Create canvas
        self.width = 500
        self.height = 500

        chatbox_width = 750
        chatbox_height = 450

        self.canvas = tk.Canvas(
            self.master, width=self.width + 800, height=self.height + 80, bg="white")

        self.chat_display = tk.Text(
            self.master, height=chatbox_height, width=chatbox_width, state=tk.DISABLED, bg="black")
        
        # Get the font used by the Text widget
        text_font = font.Font(font=self.chat_display['font'])
        # Calculate the width and height of one character in pixels
        char_width = text_font.measure('0')  # Width of a single character
        char_height = text_font.metrics('linespace')  # Height of a single line
        # Calculate the required rows and columns
        cols = int(chatbox_width / char_width)
        rows = int(chatbox_height / char_height)
        # Set the size of the Text widget
        self.chat_display.config(width=cols, height=rows)

        self.canvas.focus_set()
        self.draw()
        
        self.update_timer()

        if self.model_config:
            ExpertClasses, SolverClasses, self.model_config = load_classes_from_config(self.model_config)
            
            if len(ExpertClasses) == 1:
                self.experts = [ExpertClasses[i](role=f"EXPERT", config = self.model_config['Experts'][i]) for i in range(len(ExpertClasses))]
            else:
                self.experts = [ExpertClasses[i](role=f"EXPERT_{i + 1}", config = self.model_config['Experts'][i]) for i in range(len(ExpertClasses))]
            
            if len(SolverClasses) == 1:
                self.solvers = [SolverClasses[i](role=f"SOLVER", config = self.model_config['Solvers'][i]) for i in range(len(SolverClasses))]
            else:
                self.solvers = [SolverClasses[i](role=f"SOLVER_{i + 1}", config = self.model_config['Solvers'][i]) for i in range(len(SolverClasses))]
        else:
            self.experts = [RandomAgent(role="EXPERT")]
            self.solvers = [HumanAgent(role="SOLVER")]
            
        self.agents = self.solvers + self.experts
        
        self.chat_display.place(x=self.width + 12, y= 12 + 25)
        self.chat_display.tag_config("message", foreground="#33ff00")
        self.chat_display.tag_config("environment_title", foreground="#ffde7a")
        
        for agent in self.agents:
            if "solver" in agent.role.lower():
                self.chat_display.tag_config(f"{agent.role}_title".lower(), foreground="#3d80d9")
            else:
                self.chat_display.tag_config(f"{agent.role}_title".lower(), foreground="#d93d5a")
        
        self.solver = self.solvers[0]
        self.expert = self.experts[0]
                        
        # List of Puzzles in order they appear
        self.puzzles = []
        if self.config:
            for entry in self.config:
                self.puzzles.append(globals()[list(entry.keys())[0]])
        else:
            self.puzzles = [SimpleWirePuzzle]

        self.p_bar = tqdm(range(len(self.puzzles)))
        self.num_puzzles = len(self.puzzles)
        self.cur_puzzle = 0
        self.puzzle = self.puzzles[self.cur_puzzle]
        os.makedirs(self.save_folder, exist_ok=True)
        self.setup_puzzle()
        self.title_bar_height = self.get_title_bar_height()
                        
        while self.cur_puzzle <= len(self.puzzles):
            self.conversation_loop()



    # Initializes widgets in new puzzle and stores their location
    def setup_puzzle(self):
        self.puzzle = self.puzzles[self.cur_puzzle](self)
        solver_string = "-".join([x.__class__.__name__ for x in self.solvers])
        expert_string = "-".join([x.__class__.__name__ for x in self.experts])
        self.save_dir = os.path.join(self.save_folder, f"{solver_string}_solver_{expert_string}_expert", self.puzzle.__class__.__name__)   
         
        run_num = 0
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        elif len(os.listdir(self.save_dir)) > 0:
            runs = os.listdir(self.save_dir)
            run_numbers = [int(x.split("_")[-1]) for x in runs]
            run_num = 1 + max(run_numbers)

        self.save_dir = os.path.join(self.save_dir, f"run_{run_num}")
            
        self.canvas.update()
        if self.puzzle.buttons is not None:
            for button in self.puzzle.buttons:
                button.x_coord = button.winfo_rootx() + button.winfo_reqwidth() / 2
                button.y_coord = button.winfo_rooty() + button.winfo_reqheight() / 2
                
        for agent in self.agents:
            agent.set_module(self.puzzle)

    def get_title_bar_height(self):
        self.master.update_idletasks()  # Ensure geometry information is updated
        offset_y = 0
        if platform in ('win32', 'darwin'):
            import ctypes
            try: # >= win 8.1
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except: # win 8.0 or less
                ctypes.windll.user32.SetProcessDPIAware()
            offset_y = int(self.master.geometry().rsplit('+', 1)[-1])

        bar_height = self.master.winfo_rooty() - offset_y
        return bar_height

    # Utility function to write a message to the chat window
    def display_message(self, message, type):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{type.upper()}: ", f"{type.lower()}_title")
        self.chat_display.insert(tk.END, message + "\n", "message")
        self.chat_display.see(tk.END)  # Scroll to the last line
        self.chat_display.config(state=tk.DISABLED)
        self.master.update()

        self.write_to_conversation({"from": f"{type.upper()}", "value": message})

    # Save the conversation between agents to the outputs folder
    def write_to_conversation(self, message):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "conversation.jsonl"), "a") as f:
            f.write(json.dumps(message) + "\n")

    # Main loop between solvers and experts
    def conversation_loop(self):

        for agent in self.agents:
            agent.save_dir = self.save_dir
            
        self.cur_step = 0
        
        while self.cur_step <= self.MAX_CONVERSATION_TURNS and not self.puzzle.finished:
            
            # Initial solvers step
            for solver in self.solvers:
                image_path = solver.capture_screen_area(self.width, self.height)
                message = solver.respond(image_path, self.puzzle.actions)
                solver.step(message)
                if self.puzzle.finished:
                    break
                
            if self.puzzle.finished:
                break
            
            for expert in self.experts:
                expert_response = expert.respond(None, None)
                self.add_to_conversations({"from": expert.role, "value": expert_response})
                self.display_message(expert_response, expert.role)
        
        self.advance_puzzle(success=self.puzzle.success)

    def add_to_conversations(self, data, role=None):
        for agent in self.agents:
            if role:
                if agent.role.lower().startswith(role):
                    agent.conversation.append(data)
            else:
                agent.conversation.append(data)
            
    # Draws the boundary of the module as well as info about it such as the
    # serial number and timer
    def draw(self):
        self.canvas.create_rectangle(
            12, 12, self.width - 12, self.height - 12, outline="black", width=2, fill="gray")
        self.canvas.create_text(
            self.width // 2, self.height + 50, text=f"Serial Number: {self.serial_number}", fill="black", font=("Helvetica", 20))
        self.canvas.create_text(
            self.width + 400, 22, text=f"Solver and Expert Chat Window", fill="black", font=("Helvetica", 19))
        # Create the user input entry
        self.canvas.create_text(
            self.width + 400, self.height + 12, text=f"Chat using the console window used to run the program", fill="black", font=("Helvetica", 16))
        self.canvas.create_oval(self.width - 100, 100,
                                self.width - 50, 50, outline="black", width=2)
        self.timer_label = tk.Label(
            self.canvas, text="", font=("Arial", 20), bg="white")
        self.timer_label.place(x=self.width // 2, y=self.height + 12, anchor="center")


    # Advance to the next puzzle in the list. If success==True, it means the
    # agent solved the puzzle. Otherwise, puzzle is advancing due to the max
    # number of allowed turns being exceeded or too many mistakes.
    def advance_puzzle(self, success=True):
        self.cur_puzzle += 1
        
        if success:
            if not os.path.exists(os.path.join(self.save_dir, "conversation.jsonl")):
                os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "conversation.jsonl"), "a") as f:
                f.write(json.dumps(
                    {"from": "ENVIRONMENT", "value": "Puzzle successfully finished, moving on to the next puzzle ..."}) + "\n")
                
        score = self.puzzle.get_score()
        self.write_to_conversation(
            {"from": "ENVIRONMENT", "score": f"{score}"})

        self.cur_step = 0

        # Clear conversation history
        for agent in self.agents:
            agent.clear()

        # Clear Chatbox
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.canvas.update()

        # Clear all buttons
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button):
                widget.destroy()

        if hasattr(self.puzzle, 'stage_labels'):
            for label in self.puzzle.stage_labels:
                label.destroy()

        self.canvas.delete("all")

        self.draw()
        self.canvas.unbind("<Key>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<ButtonRelease-1>")

        if self.cur_puzzle >= len(self.puzzles):
            exit()

        self.p_bar.n = self.cur_puzzle
        self.p_bar.refresh()
        
        gc.collect()
            
        self.setup_puzzle()

    # Update the timer text every second
    def update_timer(self):
        if self.total_seconds > 0:
            minutes, seconds = divmod(self.total_seconds, 60)
            self.timer_label.config(
                text=f"Time Left: {minutes:02d}:{seconds:02d}")
            self.total_seconds -= 1
            #self.canvas.after(1000, self.update_timer)
        else:
            self.timer_label.config(text="Time's up!")


    def on_escape(self, event):
        self.canvas.focus_set()


def main(args):
    root = tk.Tk()
    app = GameManager(root, args)
    root.overrideredirect(False)
    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sound", action="store_true", default=False)
    parser.add_argument("--puzzle_config", type=str, default=None, help="Path to config file specifying the puzzle list")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--save_folder", type=str, default="./outputs")
    parser.add_argument("--baseline", type=str, default="none")
    args = parser.parse_args()
    main(args)