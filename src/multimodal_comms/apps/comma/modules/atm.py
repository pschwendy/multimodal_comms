from multimodal_comms.apps.comma.modules.module import Module
from multimodal_comms.apps.comma.modules.util import HighlightButton
import random
import functools

class AtmPuzzle(Module):
    """The solver will be presented with a bank interface. Their goal is to deposit $100 if they have less than $500
    in their balance, or withdraw $300 if they have at least $500 dollars. All transactions and balance information
    is protected behind a 4-digit PIN number which only the solver has access to.
    """
    def __init__(self, game_manager):
        super(AtmPuzzle, self).__init__(game_manager)

        self.actions = {}
        action_names = ["press_0", "press_1", "press_2", "press_3", "press_4", "press_5", "press_6", "press_7", "press_8", "press_9", "press_up_arrow", "press_down_arrow", "press_enter", "press_delete"]
        for action in action_names:
            self.actions[action] = functools.partial(self.press_button, None, action.split("_")[1])
            
        self.user_entered_PIN = ""
        self.transaction_amount = ""
        self.transaction_type = "DEPOSIT"

        if self.game_manager.config: # If you have some config for the puzzle, you can load it here
            my_config = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]
            self.ground_truth_PIN = my_config["PIN"]
            self.balance = my_config['Balance']
        else: # Some initialization logic
            self.balance = random.randint(1, 1000)
            self.ground_truth_PIN = str(random.randint(1000,9999))
            
        self.solver_private_info = f"The PIN number for your bank account is {self.ground_truth_PIN}"

        # Create some buttons / canvas elements if you like
        # This defines a button, which has a handler called on_submit()
        button_width = 60
        button_height = 40
        button_gap = 10
        start_x = 25
        start_y = 250

        positions = [
            (0, 0), (1, 0), (2, 0),  # Row 1: 1, 2, 3
            (0, 1), (1, 1), (2, 1),  # Row 2: 4, 5, 6
            (0, 2), (1, 2), (2, 2),  # Row 3: 7, 8, 9
            (1, 3)                   # Row 4: 0
        ]

        for i, (col, row) in enumerate(positions):
            x = start_x + col * (button_width + button_gap)
            y = start_y + row * (button_height + button_gap)
            button_text = str(i + 1) if i < 9 else "0"

            button = HighlightButton(
                self,
                x,
                y,
                button_width,
                button_height,
                text= button_text, # Map index to numbers
                command=functools.partial(self.game_manager.solver.step, f"press_{button_text}")
            )

            # When adding buttons, make sure to add them to self.buttons
            self.buttons.append(button)

        enter_button = HighlightButton(
                self,
                275,
                300,
                100,
                40,
                text= "ENTER", # Map index to numbers
                command=functools.partial(self.game_manager.solver.step, f"press_enter")
            )

        up_button = HighlightButton(
                self,
                275,
                250,
                40,
                40,
                text= "↑", # Map index to numbers
                command=functools.partial(self.game_manager.solver.step, f"press_up_arrow")
            )
        
        down_button = HighlightButton(
                self,
                335,
                250,
                40,
                40,
                text= "↓", # Map index to numbers
                command=functools.partial(self.game_manager.solver.step, f"press_down_arrow")
            )
        
        del_button = HighlightButton(
                self,
                165,
                400,
                60,
                40,
                text= "DEL", # Map index to numbers
                command=functools.partial(self.game_manager.solver.step, f"press_delete")
            )


        # When adding buttons, make sure to add them to self.buttons
        self.buttons.append(enter_button)
        self.buttons.append(del_button)
        self.buttons.append(up_button)
        self.buttons.append(down_button)

        self.canvas.create_rectangle(start_x, 25, 375, start_y - 25, outline="black", fill="black", width=2)
        

        welcome_blurb = self.canvas.create_text(200, 35, text="Welcome to COMMA Bank", fill="#33ff00", font=("Helvetica", 12))
        transaction_blurb = self.canvas.create_text(200, 90, text="Please Select Transaction Type:", fill="#33ff00", font=("Helvetica", 12))
        self.pointer = self.canvas.create_text(100, 120, text="→", fill="#f2f542", font=("Helvetica", 24))
        deposit_option = self.canvas.create_text(200, 120, text="Deposit", fill="#f2f542", font=("Helvetica", 12))
        withdraw_option = self.canvas.create_text(200, 150, text="Withdrawal", fill="#33ff00", font=("Helvetica", 12))

        self.transaction_page = [[welcome_blurb, transaction_blurb], [deposit_option, withdraw_option]]
        
        deposit_blurb = self.canvas.create_text(200, 90, text="Making a Deposit...", fill="#33ff00", font=("Helvetica", 12))
        pin_message = self.canvas.create_text(200, 120, text="Enter your PIN: ", fill="#33ff00", font=("Helvetica", 12))
        self.pin_text = self.canvas.create_text(300, 120, text="", fill="#ffffff", font=("Helvetica", 12))
        back_button_1 = self.canvas.create_text(200, 150, text="Go Back", fill="#33ff00", font=("Helvetica", 12))

        self.deposit_page = [[deposit_blurb, self.pin_text],[pin_message, back_button_1]]

        withdraw_blurb = self.canvas.create_text(200, 90, text="Making a Withdrawal...", fill="#33ff00", font=("Helvetica", 12))
        pin_message2 = self.canvas.create_text(200, 120, text="Enter your PIN: ", fill="#33ff00", font=("Helvetica", 12))
        back_button_2 = self.canvas.create_text(200, 150, text="Go Back", fill="#33ff00", font=("Helvetica", 12))

        self.withdraw_page = [[withdraw_blurb, self.pin_text],[pin_message2, back_button_2]]

        transaction_type_blurb = self.canvas.create_text(200, 35, text="", fill="#33ff00", font=("Helvetica", 12))
        balance_blurb = self.canvas.create_text(200, 90, text=f"Your Current Balance is ${self.balance}", fill="#33ff00", font=("Helvetica", 12))
        transaction_amount = self.canvas.create_text(200, 120, text="Enter Amount: ", fill="#33ff00", font=("Helvetica", 12))
        self.transaction_amount_text = self.canvas.create_text(300, 120, text="", fill="#ffffff", font=("Helvetica", 12))
        back_button_3 = self.canvas.create_text(200, 150, text="Go Back", fill="#33ff00", font=("Helvetica", 12))

        self.final_page = [[transaction_type_blurb, balance_blurb, self.transaction_amount_text],[transaction_amount, back_button_3]]

        self.active_page = self.transaction_page
        self.selected_option_idx = 0

        self.deactivate_page(self.withdraw_page)
        self.deactivate_page(self.deposit_page)
        self.deactivate_page(self.final_page)

        self.page_mapping = {
            "Transaction": self.transaction_page,
            "Deposit": self.deposit_page,
            "Withdrawal": self.withdraw_page,
        }

        self.prev_page = None
    
    def activate_page(self, page):
        self.selected_option_idx = 0
        for element in page[0] + page[1]:
            self.canvas.itemconfig(element, state='normal')
        x, _ = self.canvas.coords(self.pointer)
        _, option_y = self.canvas.coords(page[1][self.selected_option_idx])
        self.canvas.coords(self.pointer, x, option_y)
        self.canvas.itemconfig(page[1][self.selected_option_idx], fill="#f2f542")  # Update color



    def deactivate_page(self, page):
        for element in page[0] + page[1]:
            self.canvas.itemconfig(element, state='hidden')
            self.canvas.itemconfig(element, fill="#33ff00")  # Update color


    def switch_to_page(self, page):
        self.prev_page = self.active_page
        self.deactivate_page(self.active_page)
        self.active_page = page
        self.activate_page(self.active_page)

    def check_correct(self):
        if len(self.transaction_amount) == 0:
            return False
        
        amount = int(self.transaction_amount)

        if self.balance >= 500 and self.transaction_type == "Withdrawal" and amount == 300:
            return True
        elif self.balance < 500 and self.transaction_type == 'Deposit' and amount == 100:
            return True
        return False
    
    def press_button(self, event=None, text="Back"):
        if text == "up":
            old_idx = self.selected_option_idx
            self.selected_option_idx = max(self.selected_option_idx - 1, 0)
            if self.selected_option_idx != old_idx:
                self.canvas.itemconfig(self.active_page[1][old_idx], fill="#33ff00")  # Update color
                self.canvas.itemconfig(self.active_page[1][self.selected_option_idx], fill="#f2f542")  # Update color

            x, _ = self.canvas.coords(self.pointer)
            _, option_y = self.canvas.coords(self.active_page[1][self.selected_option_idx])
            self.canvas.coords(self.pointer, x, option_y)
        elif text == "down":
            old_idx = self.selected_option_idx
            self.selected_option_idx = min(self.selected_option_idx + 1, len(self.active_page[1]) - 1)
            if self.selected_option_idx != old_idx:
                self.canvas.itemconfig(self.active_page[1][old_idx], fill="#33ff00")  # Update color
                self.canvas.itemconfig(self.active_page[1][self.selected_option_idx], fill="#f2f542")  # Update color

            x, _ = self.canvas.coords(self.pointer)
            _, option_y = self.canvas.coords(self.active_page[1][self.selected_option_idx])
            self.canvas.coords(self.pointer, x, option_y)
        elif text == "enter":
            page_name = self.canvas.itemcget(self.active_page[1][self.selected_option_idx], "text")
            if page_name == "Go Back":
                # Clear PIN
                self.user_entered_PIN = ""
                self.canvas.itemconfig(self.pin_text, text="")

                self.switch_to_page(self.page_mapping['Transaction'])
            elif page_name.startswith("Enter your PIN"):
                if self.user_entered_PIN == self.ground_truth_PIN:
                    self.switch_to_page(self.final_page)
                    self.canvas.itemconfig(self.active_page[0][0], text=f"Making a {self.transaction_type}...")
                else:
                    self.log_mistake()
            elif page_name.startswith("Enter Amount"):
                self.finish_check()
            else:
                # Clear PIN
                self.user_entered_PIN = ""
                self.canvas.itemconfig(self.pin_text, text="")
                if page_name in ['Deposit', 'Withdrawal']:
                    self.transaction_type = page_name
                self.switch_to_page(self.page_mapping.get(page_name, "Transaction"))
        elif text.isdigit() or text == 'delete':
            t = self.canvas.itemcget(self.active_page[1][self.selected_option_idx], "text")
            if t.startswith("Enter your PIN"):
                if text == 'delete' and len(self.user_entered_PIN) > 0:
                    self.user_entered_PIN = self.user_entered_PIN[:len(self.user_entered_PIN) - 1]
                    self.canvas.itemconfig(self.pin_text, text="*" * len(self.user_entered_PIN))
                elif text.isdigit() and len(self.user_entered_PIN) < 4:
                    self.user_entered_PIN += text
                    self.canvas.itemconfig(self.pin_text, text="*" * len(self.user_entered_PIN))
            elif t.startswith("Enter Amount"):
                if text == 'delete' and len(self.transaction_amount) > 0:
                    self.transaction_amount = self.transaction_amount[:len(self.transaction_amount) - 1]
                    self.canvas.itemconfig(self.transaction_amount_text, text=self.transaction_amount)
                elif text.isdigit() and len(self.transaction_amount) < 5:
                    self.transaction_amount += text
                    self.canvas.itemconfig(self.transaction_amount_text, text=self.transaction_amount)

    # String representation of the puzzle
    def __str__(self):
        return "AtmPuzzle"
