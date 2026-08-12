import itertools
import json

# Define the display_text and button_text lists.
display_text_list = [1, 2, 3, 4]
button_text_list = ["1", "2", "3", "4"]

# Generate all button_text permutations.
button_permutations = list(itertools.permutations(button_text_list))

# Combine every display_text value with every button_text permutation.
all_combinations = []
for display_text in display_text_list:
    for button_texts in button_permutations:
        scenario = {
            "display_text": display_text,
            "button_texts": button_texts
        }
        all_combinations.append(scenario)

# Write every generated case to a JSON file.
with open('memory.json', 'w') as json_file:
    json.dump(all_combinations, json_file, indent=4)
