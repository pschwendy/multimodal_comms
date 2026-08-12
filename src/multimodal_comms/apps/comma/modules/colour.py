from multimodal_comms.apps.comma.modules.module import Module
import random
import functools
from multimodal_comms.apps.comma.modules.util import HighlightButton

class ColourPuzzle(Module):
    """he solver presses squares of the least common color in a 4x4 grid,
    then follows a sequence based on a table, aiming to turn all squares white.
    """
    def __init__(self, game_manager):
        super(ColourPuzzle, self).__init__(game_manager)

        
        self.colors = ["Red", "Blue", "Green",
                       "Yellow", "Magenta", "Row", "Column"]
        self.color_to_index = {color: idx for idx,
                               color in enumerate(self.colors)}
        self.grid_size = 4
        self.actions = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.actions[f"click_row_{i}_col_{j}"] = functools.partial(self.on_button_click, None, i, j)
        self.previous_group = None
        self.buttons = []
        self.create_widgets()
        self.reset_module()
        self.max_score = 0

    def create_widgets(self):
        self.setup_buttons()

    def setup_buttons(self):
        self.buttons = []
        button_size = 60
        start_x = (self.width - button_size * self.grid_size) // 2
        start_y = (self.height - button_size * self.grid_size) // 2

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                button = HighlightButton(self, start_x + j * button_size, start_y + i * button_size, button_size,
                                         button_size, text="", command=functools.partial(self.execute_action, f"click_row_{i}_col_{j}"))
                self.buttons.append(button)

    def reset_module(self):
        self.previous_group = None
        self.randomize_colors()

    def randomize_colors(self):
        if self.game_manager.config:
            for i, button in enumerate(self.buttons):
                color = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]["colors"][i]
                button.config(bg=color)
        else:
            for button in self.buttons:
                color = random.choice(self.colors[:5])
                button.config(bg=color)
            
        non_white = [btn for btn in self.buttons if btn.cget("bg") != "White"]

        for i, color in enumerate(self.colors[:5]):
            button = non_white[i]
            curr = button.cget("bg")
            button.config(bg=color)

    def get_group_to_press(self):
        color_counts = {color: 0 for color in self.colors[:5]}
        for button in self.buttons:
            color = button.cget("bg")
            if color in color_counts:
                color_counts[color] += 1

        min_count = min(color_counts.values())
        min_colors = [color for color,
                      count in color_counts.items() if count == min_count]
        if len(min_colors) == 1:
            return min_colors[0]
        else:
            # Implement additional logic if needed for ties
            return min_colors[0]

    def determine_next_group(self):
        if not self.previous_group:
            return self.get_group_to_press()

        current_white_squares = sum(button.cget(
            "bg") == "White" for button in self.buttons)
        current_row = current_white_squares
#[\"Red\", \"Blue\", \"Green\", \"Yellow\", \"Magenta\", \"Row\", \"Column\"]
        table = [
            ["Blue", "Column", "Red", "Yellow", "Row", "Green", "Magenta"],
            ["Row", "Green", "Blue", "Magenta", "Red", "Column", "Yellow"],
            ["Yellow", "Magenta", "Green", "Row", "Blue", "Red", "Column"],
            ["Blue", "Green", "Yellow", "Column", "Red", "Row", "Magenta"],
            ["Yellow", "Row", "Blue", "Magenta", "Column", "Red", "Green"],
            ["Magenta", "Red", "Yellow", "Green", "Column", "Blue", "Row"],
            ["Green", "Row", "Column", "Blue", "Magenta", "Yellow", "Red"],
            ["Magenta", "Red", "Green", "Blue", "Yellow", "Column", "Row"],
            ["Column", "Yellow", "Red", "Green", "Row", "Magenta", "Blue"],
            ["Green", "Column", "Row", "Red", "Magenta", "Blue", "Yellow"],
            ["Red", "Yellow", "Row", "Column", "Green", "Magenta", "Blue"],
            ["Column", "Row", "Column", "Row", "Row", "Column", "Row"],
            ["Row", "Column", "Row", "Column", "Row", "Column", "Column"],
            ["Column", "Column", "Row", "Row", "Column", "Row", "Column"],
            ["Row", "Row", "Column", "Row", "Column", "Column", "Row"]
        ]

        group_index = self.color_to_index.get(self.previous_group, -1)
        if group_index == -1:
            return None

        return table[current_row - 1][group_index]

    def on_button_click(self, event=None, i=0, j=0):
        button = self.buttons[i * self.grid_size + j]
        color = button.cget("bg")
        if color == "White":
            return

        group_to_press = self.determine_next_group()
        if group_to_press and (color == group_to_press or self.is_in_group(group_to_press, i, j)):
            self.press_group(group_to_press, i, j)

        else:
            white = len([btn for btn in self.buttons if btn.cget("bg") == "White"])
            self.max_score = max([self.max_score, white / (self.grid_size * self.grid_size)])
            self.log_mistake()
            self.reset_module()

    def is_in_group(self, group, i, j):
        if group == "Row":
            # Check if the current row is the uppermost row with non-white squares
            for row in range(self.grid_size):
                if any(self.buttons[row * self.grid_size + k].cget("bg") != "White" for k in range(self.grid_size)):
                    return row == i
            return False
        elif group == "Column":
            # Check if the current column is the leftmost column with non-white squares
            for col in range(self.grid_size):
                if any(self.buttons[k * self.grid_size + col].cget("bg") != "White" for k in range(self.grid_size)):
                    return col == j
            return False
        return False

    def press_group(self, group, row=None, col=None):
        if group in self.colors[: 5]:
            for button in self.buttons:
                if button.cget("bg") == group:
                    button.config(bg="White")
        elif group == "Row":
            for i in range(self.grid_size):
                self.buttons[row * self.grid_size + i].config(bg="White")
        elif group == "Column":
            for i in range(self.grid_size):
                self.buttons[i * self.grid_size + col].config(bg="White")

        if all(button.cget("bg") == "White" for button in self.buttons):
            self.finish_check()
        else:
            self.previous_group = group
            self.update_colors()

    def update_colors(self):
        for button in self.buttons:
            if button.cget("bg") != "White":
                button.config(bg=random.choice(self.colors[: 5]))

        non_white = [btn for btn in self.buttons if btn.cget("bg") != "White"]
        white = len([btn for btn in self.buttons if btn.cget("bg") == "White"])
        self.max_score = max([self.max_score, white / (self.grid_size * self.grid_size)])

        if len(non_white) >= 5:
            for i, color in enumerate(self.colors[:5]):
                button = non_white[i]
                curr = button.cget("bg")
                button.config(bg=color)

    def check_correct(self):
        # return True
        return all(button.cget("bg") == "White" for button in self.buttons)

    def get_score(self):
        return self.max_score

    def __str__(self):
        return "ColourPuzzle"
