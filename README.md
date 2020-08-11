# Tennis
Solution to the third project of Udacitys Reinforcement Learning Nanodegree


### Quick Installation

To set up the python environment and install all requirements for using this repository on Linux OS, follow the instructions given below:
1. Create and activate a new environment with Python 3:
    ```bash
    python3 -m venv /path/to/virtual/environment
    source /path/to/virtual/environment/bin/activate
    ```
2. Clone the Udacity repository and navigate to the `python/` folder to install the `unityagents`-package:
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    cd -
    ```
3. Download the Unity-environment from Udacity and unzip it:
    ```bash
    wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
    unzip Tennis_Linux.zip
    rm Tennis_Linux.zip
    ```

For using this repository using Windows OS or Mac OSX, please follow the instructions given ![here](https://github.com/udacity/deep-reinforcement-learning#dependencies). Afterwards download the Unity-environment for ![Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip), ![Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip) or ![Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip) and unzip it in this repository.



### Quick Start

After installing all requirements and activating the virtual environment training the agent can be started by executing

```bash
python main.py
```

Configurations like enabling the visualization or adjustments to the architecture is possible through the dictionaries defined in `main.py`.
During training a file called `performance.log` is created, which holds information about the current score, the average score of the last 100 episodes, the current loss of the neural network and the average loss of the last 100 episodes. Furthermore, if the agents variable `save_after` is set to a value larger than 0, after the given number of epochs the agents current parameters will be saved in a file with the agents name plus `_parameters.dat`, while the current weights of the agents models will be saved in a file with the agents name plus `_decision.model` or `_policy.model` respectively.

Running

```bash
python evaluate.py
```

allows to evaluate the performance of the saved agent of the given name. 

