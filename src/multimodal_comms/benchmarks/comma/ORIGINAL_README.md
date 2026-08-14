#  COMMA : A Communicative Multimodal Multi-Agent Benchmark

![Local Image](./assets/agents.jpg)

COMMA is a novel benchmark designed to evaluate the collaborative performance of multimodal multi-agent systems through language communication.

We assess multi-modal multi-agent systems using a series of carefully designed collaborative puzzle
games. These scenarios typically involve two-player setups where agents have access to different,
complementary information.

Our benchmark features over 10 customizable puzzles with thousands of solutions. We assessed AI-AI and AI-Human settings, testing popular closed-source (o4-mini, GPT-4o, GPT-4V, Gemini) and open-source (QwenVL, InternVL, LLaVA, LLaMA 3.2) multimodal models. Notably, the open-source models often did not surpass a basic random baseline in the AI-AI scenario, indicating large room for improvement.

## 🚀Quickstart

Run one of the following scripts depending on your operating system to setup our environment and download the PAD_UFES images. Note, we do not support Mac yet due to issues with rendering and libraries not compiling. We will look to support it in the near future!

### Installation for Windows
    setup_windows.ps1

### Installation for Linux
    bash setup.sh

The code is structured as displayed in the image below:

![Local Image](./assets/code_diagram.jpg)

To evaluate model predictions on COMMA, you need to specify which puzzles to evaluate on (the `--puzzle_config` argument), as well as which Solver and Expert agents you are evaluating (the `--model_config` argument). Both are just .json files structured as explained below.

You can specify the Solver and Expert agents in `./config/experiment_config.json`. We also have 2 filled in example config files in `./config/random_config.json` and `./config/human_config.json`, along with more in `./config/experiments_AI`:

```
{
     "Hyperparameters": {
        "MAX_MISTAKES": 3,
        "MAX_CONVERSATION_TURNS": 20,
        "SERIAL_NUMBER": 135790,
        "TOTAL_TIME": 3000
    },
    "Experts": [{
        "file_path": "agents/gpt4o_agent.py",
        "class_name": "GPT4oAgent",
        "API_KEY": "<Your API Key Here>",
        "API_VERSION": "2023-12-01-preview",
        "API_BASE": "https://chatgpt-simulation.openai.azure.com/" 
    }],
    "Solvers": [{
        "file_path": "agents/gpt4o_agent.py",
        "class_name": "GPT4oAgent",
        "API_KEY": "<Your API Key Here>",
        "API_VERSION": "2023-12-01-preview",
        "API_BASE": "https://chatgpt-simulation.openai.azure.com/" 
    }]
}
```

If you are using proprietary models with an azure API such as GPT4o, GPT4v, etc, make sure to put your API key in `config/experiment_config.json`.

Next, make sure you have a `puzzles.json` file which contains details about the puzzles you would like to evaluate on. We have some examples in the `config` folder. This file should just be a json file which is a list of puzzle modules like this:

```
[
    {
        "AtmPuzzle": {
            "notes": "Need to specify PIN number and Balance.",
            "PIN": "3285",
            "Balance": 600
        }
    },
    {
        "SimpleWirePuzzle": {
            "notes": "Options for n_wires are 3-6. Colors should be of length n_wires, and each color may be one of red, white, yellow, blue, black",
            "n_wires": 4,
            "colors": ["blue", "black", "yellow", "red"]
        }
    }
    ⋮
]

```

If you are working remotely (e.g. on a linux server), please refer to the next section for how to setup and run with xvfb. However, if you are working locally on windows, you can run the following command.
```
python main.py --puzzle_config ./config/puzzles_final.json --model_config ./config/random_config.json
```

### Test on Remote Server

This section aims to help you run the experiment on your remote server, especially when it's not with a GUI. 

1. **Install Docker**. Follow the instructions in the [Docker setup guide](https://docs.docker.com/engine/install/) to install Docker on your machine. 
2. **Enter a Docker Container**.
    ```
    docker run -it --rm -p5900:5900 ubuntu:20.04
    ```
3. **Install the X component**.
    ```
    apt update
    apt install -y xserver-xorg
    apt install xvfb
    apt install x11vnc
    ```
4. **Run script with a virtual screen**. For instance:
    ```
    sudo xvfb-run -n 2 -e /dev/stdout python -u main.py --puzzle_config config/puzzles_final.json --model_config config/random_config.json
    ```
5. (Optional) Use a VNC server to see the screen.
   1) Open a new terminal and run `ps -ef |grep auth`.
    Then we find the location of Auth file:
    ```
    root@13785a282294:/# ps -ef |grep auth
    root        7417    7408  1 11:47 pts/0    00:00:00 Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp -auth /tmp/xvfb-run.RCwemo/Xauthority
    root        7449    5837  0 11:47 pts/1    00:00:00 grep --color=auto auth
    ```
    `/tmp/xvfb-run.RCwemo/Xauthority` is the path of Auth file, which is generated randomly for each time.

    `:99` is the screen number of the virtual screen. It is default to be 99.

   2) **Start the vnc server**.
   ```
   x11vnc -display :99 -auth /tmp/xvfb-run.RCwemo/Xauthority #Replace the path with your Auth file
   ```
   x11vnc listens on port `5900` by default.

   3) **Using a VNC client e.g. `TightVNC`, `RealVNC Viewer` to see the screen**.

### Deploy your models
Use `agents\template.py` as a generic agent template to test your own models on COMMA.

### Come up with your own tasks
Our benchmark also allows for users to add their own tasks. To do so, follow these steps:

1. Create a new file in modules, and fill it out according to the template described in `modules/module_template.py`

2. Add the instructions for solving the puzzle in `config/puzzles.json`

3. (Optional) Add an image for the puzzle manual for the expert in `images/manuals`

4. Import your module in `modules/__init__.py`

5. Either add your puzzle to `config/puzzles.json` or manually in `main.py`

### Summarize Results After Experiments
By default, running the experiments will save the conversations between agents to a folder called outputs. You can summarize the results based on the conversations in an output folder with the following command:

```
python summarize_results.py --result_folder <path_to_your_folder_containing_agent_conversations>
```
We also provide the final conversations used in our analysis in the folder called `final_results`. You can reproduce the figures in our paper by running the `summarize_results.py` script with this folder as an argument.
