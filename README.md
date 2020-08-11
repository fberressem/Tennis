# Tennis
Solution to the third project of Udacitys Reinforcement Learning Nanodegree

![video](https://user-images.githubusercontent.com/63595824/89938604-05547100-dc17-11ea-8e90-067b6e4440f5.gif)

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

### Background Information
In this environment two players (or agents) are playing a game of tennis. The aim for both agents is to collect as many points as possible, where a player gets a reward of **`+0.1`** every time he manages to play the ball over the net and a reward of **`-0.01`** if the player lets the ball hit the ground or hits it out of bounds and the episode is ended. Hence, both players try to collaborate while playing by keeping the ball in the game for as long as possible to maximize their reward. At every timestep, each player is provided a stack of 3 8-dimensional observations and has to decide on the actions to take. For every player, the corresponding action-vector consists of 2 real numbers in the range from -1 to +1. The game is ended after one of the players drops the ball or hits it out of bounds or after a given number of successful passes, which makes this game an episodic one. In every episode, the score is calculated as the maximum summed up rewards of the player obtained during this episode. Explicitely, this means that the score of episode *i* is defined as

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?Score_i=max\left(\sum_{r\in\;R^A_i}r,\sum_{r\in\;R^B_i}r\right)" /></p>

where <img src="https://latex.codecogs.com/svg.latex?R^A_i"> is the set of rewards obtained by player *A* during episode *i*, and <img src="https://latex.codecogs.com/svg.latex?R^B_i"> for player *B* correspondingly.

The environment is considered solved, when the average (over 100 episodes) of those scores defined above is at least **`+0.5`**.

For more information on the approach that was used to solve this environment, see [`Report.md`](https://github.com/fberressem/Tennis/blob/master/Report.md).
