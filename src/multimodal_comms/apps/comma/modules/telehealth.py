from multimodal_comms.apps.comma.modules.module import Module
from PIL import Image, ImageTk
import random
import pandas as pd
import functools
import os
from multimodal_comms.apps.comma.paths import IMAGE_ROOT
import math


class TelehealthPuzzle(Module):
    """The solver plays the role of a bystander in a health crisis scene.
    The expert plays the role of a telehealth operator, offering the solver
    instructions on how to address the situation.
    """
    def __init__(self, game_manager):
        super(TelehealthPuzzle, self).__init__(game_manager)

        self.best_score = 0
        self.stage = 0
        start_x = 25
        start_y = 500
        self.canvas.create_rectangle(start_x, 25, 375, start_y - 25, outline="black", fill="black", width=2)
        welcome_blurb = self.canvas.create_text(200, 35, text="Welcome to COMMA Health", fill="#33ff00", font=("Helvetica", 12))
        instruction = self.canvas.create_text(200, 90, text="First diagnose your skin lesion", fill="#33ff00", font=("Helvetica", 10))
        instuction_2 = self.canvas.create_text(200, 120, text="Then select the appropriate treatment", fill="#33ff00", font=("Helvetica", 10))
        
                
        data = pd.read_csv("./modules/metadata.csv")
        
        if self.game_manager.config:
            config = self.game_manager.config[self.game_manager.cur_puzzle][str(self)]
            self.idx = config.get("skin_lesion_idx", 0)
        else:
            self.idx =  random.randint(0, len(data) - 1)
        
        self.patient_info = dict(data.iloc[self.idx, :])
        dataset_root = os.environ.get("COMMA_PAD_UFES_DIR", str(IMAGE_ROOT / "PAD_UFES" / "images"))
        path = os.path.join(dataset_root, self.patient_info['img_id'])
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Telehealth images are not bundled; set COMMA_PAD_UFES_DIR to the PAD-UFES image directory"
            )
        pil_image = Image.open(path)
        pil_image = pil_image.resize((300,300))
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_id = self.canvas.create_image(200, 290, image = tk_image, anchor='center')
        self.canvas.image = tk_image
        
        def summarize_patient(profile):
            """
            Generate a summary for a patient profile, ignoring values that are nan.
            
            Args:
                profile (dict): A dictionary containing patient information.
                
            Returns:
                str: A summary string.
            """
            summary = []
            invalid_keys = ["patient_id", "lesion_id", "img_id", "diagnostic"]
            for key, value in profile.items():
                if key in invalid_keys:
                    continue
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    if isinstance(value, bool):  # Convert boolean to readable format
                        value = "Yes" if value else "No"
                    summary.append(f"{key.replace('_', ' ').capitalize()}: {value}")
            
            return "\n".join(summary)

        self.solver_private_info = f"Here is your patient profile:\n{summarize_patient(self.patient_info)}"
        
        self.actions = {
            "diagnose_BCC": functools.partial(self.check_action, "diagnose_BCC"),
            "diagnose_SCC": functools.partial(self.check_action, "diagnose_SCC"),
            "diagnose_MEL": functools.partial(self.check_action, "diagnose_MEL"),
            "diagnose_ACK": functools.partial(self.check_action, "diagnose_ACK"),
            "diagnose_NEV": functools.partial(self.check_action, "diagnose_NEV"),
            "diagnose_SEK": functools.partial(self.check_action, "diagnose_SEK"),
            "treat_surgery": functools.partial(self.check_action, "treat_surgery"),
            "treat_no_treatment": functools.partial(self.check_action, "treat_no_treatment"),
            "treat_cryotherapy": functools.partial(self.check_action, "treat_cryotherapy"),
        }
   
    def check_action(self, action_name):
        if self.stage == 0:
            prediction = action_name.split("_")[1]
            if self.patient_info['diagnostic'].replace('BOD', 'SCC') == prediction:
                self.stage = 1
                self.best_score = 0.5
                return
            self.log_mistake()
        elif self.stage == 1:
            treatment = action_name.split("_", 1)[1]
            gt_treatment = None
            diagnosis = self.patient_info['diagnostic'].replace('BOD', 'SCC')
            if diagnosis == 'SEK':
                if self.patient_info.get('itch', False) or self.patient_info.get('bleed', False):
                    gt_treatment = "cryotherapy"
                else:
                    gt_treatment = "no_treatment"
            elif diagnosis == "BCC" or diagnosis == "SCC":
                if self.patient_info.get('diameter_1', 0) >= 5.0 or self.patient_info.get('diameter_2', 0) >= 5.0:
                    gt_treatment = "surgery"
                else:
                    gt_treatment = "cryotherapy"
            elif diagnosis == "MEL":
                gt_treatment = "surgery"
            elif diagnosis == "NEV":
                gt_treatment = "no_treatment"
            elif diagnosis == "ACK":
                gt_treatment = "cryotherapy"
            
            if gt_treatment == treatment:
                if self.finish_check().startswith("The puzzle is solved!"):
                    self.canvas.delete(self.image_id)
                return
                
            self.log_mistake()
   
    def check_correct(self):
        self.best_score = 1
        return True
    
    def get_score(self):
        return self.best_score

    def __str__(self):
        return "TelehealthPuzzle"
