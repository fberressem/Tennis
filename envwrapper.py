from unityagents import UnityEnvironment


class Env():
    """ Environment for multiple agents to interact with """
    def __init__(self, no_graphics = True):
        """ Initialize Env object
        
        Params
        ======
            no_graphics(bool): flag for visualizing interaction with environment
        """
        self.env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64", no_graphics = no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # reset the environment
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]

        # number of agents in the environment
        print('Number of agents:', len(self.env_info.agents))

        # number of actions
        self.action_size = self.brain.vector_action_space_size
        print('Number of actions:', self.action_size)

        # examine the state space 
        state = self.env_info.vector_observations[0]
        print('States look like:', state)
        self.state_size = len(state)
        print('States have length:', self.state_size)

    def return_sizes(self):
        """ Returns action, state dimensions and number of agents """
        return self.action_size, self.state_size, len(self.env_info.agents)

    def reset(self):
        """ Resets environment and returns start state """
        self.env_info = self.env.reset(train_mode=True)[self.brain_name] # reset the environment
        state = self.env_info.vector_observations            # get the current state
        return state

    def step(self, action):
        """ Returns next state, reward and whether the next state is a terminal state

        Params
        ======
            action(array): actions chosen by agents
        """
        self.env_info = self.env.step(action)[self.brain_name]
        next_state = self.env_info.vector_observations
        reward = self.env_info.rewards
        done = self.env_info.local_done
        return next_state, reward, done, []

    def close(self):
        """ Closes environment """
        self.env.close()

