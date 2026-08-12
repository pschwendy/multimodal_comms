
import itertools
import json

n_wires = [3, 4, 5, 6]
colors = ["red", "white", "yellow", "blue", "black"]

def generate_combinations(wire_count, colors):
    return list(itertools.product(colors, repeat=wire_count))

all_combinations = []
for wire_count in n_wires:
    color_combinations = generate_combinations(wire_count, colors)
    for combination in color_combinations:
        scenario = {
            "wire_count": wire_count,
            "colors": combination
        }
        all_combinations.append(scenario)

with open('wire_color_combinations.json', 'w') as json_file:
    json.dump(all_combinations, json_file, indent=4)