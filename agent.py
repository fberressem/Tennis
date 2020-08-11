import torch
import torch.nn as nn

import torch.optim as optim

import numpy as np

from collections import deque

import utils
import model
import time

class Agent():
    """ Agent to interact with environment"""
    def __init__(self, agent_dict={}, actor_dict={}, critic_dict={}):
        """ Initialize Agent object

        Params
        ======
            agent_dict(dict): dictionary containing parameters for agent
            actor_dict(dict): dictionary containing parameters for agents actor-model
            critic_dict(dict): dictionary containing parameters for agents critic-model
        """
        enable_cuda = agent_dict.get("enable_cuda", False)
        if enable_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.num_agents = agent_dict.get("num_agents", 2)
        self.num_actions = agent_dict.get("num_actions", 2)        
    
        self.num_episodes = agent_dict.get("num_episodes", 10000)
        self.save_after = agent_dict.get("save_after", -1)
        self.name = agent_dict.get("name", "tennis")

        self.gamma = agent_dict.get("gamma", 0.9)

        self.tau = agent_dict.get("tau", 0.001)

        self.noise = utils.OUNoise((self.num_agents, self.num_actions), 0)

        self.num_replays = agent_dict.get("num_replays", 1)

        self.learning_rate_actor = agent_dict.get("learning_rate_actor", 1E-3)
        self.learning_rate_critic = agent_dict.get("learning_rate_critic", 1E-3)

        self.criterion = nn.MSELoss()

        memory_size = agent_dict.get("memory_size", 2**14)
        batchsize = agent_dict.get("batchsize", 2**10)
        replay_reg = agent_dict.get("replay_reg", 0.0)

        self.replay_buffer = utils.ReplayBuffer(memory_size, batchsize, self.device)

        self.positive_emphasis = agent_dict.get("positive_emphasis", False)
        
        if self.positive_emphasis:
            self.positive_replay_buffer = utils.ReplayBuffer(memory_size, batchsize, self.device)


        self.actor = model.ActorModel(actor_dict).to(self.device)
        self.actor_target = model.ActorModel(actor_dict).to(self.device)
        
        self.critic = model.CriticModel(critic_dict).to(self.device)
        self.critic_target = model.CriticModel(critic_dict).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)


        utils.copy_model(self.actor, self.actor_target, tau=1.0)
        utils.copy_model(self.critic, self.critic_target, tau=1.0)
    
        seed = agent_dict.get("seed", 0)

        torch.manual_seed(seed)
        np.random.seed(seed)


    def preprocess(self, state):
        """ Convert state to torch.Tensor

        Params
        ======
            state(array): current state
        """
        return torch.from_numpy(state).float().to(self.device)

    def act(self, states, add_noise=True):
        """ Choose action according to actor. Returns actions and states as torch.Tensor

        Params
        ======
            state(array): current state
            add_noise(bool): flag to add noise to action
        """
        states_t = self.preprocess(states)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states_t).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            actions += self.noise.sample()
        return actions, states_t


    def learn(self, positive=False):
        """ Train actor and critic based on past experiences """
        for t in range(self.num_replays):
            if not positive:
                states_t, actions_t, rewards_t, next_states_t, dones_t = self.replay_buffer.sample()
            else:
                if len(self.positive_replay_buffer) == 0:
                    return
                states_t, actions_t, rewards_t, next_states_t, dones_t = self.positive_replay_buffer.sample()


            next_actions_t = self.actor_target(next_states_t)
            next_q_values_t = self.critic_target(next_states_t, next_actions_t).detach()

            targets_t = rewards_t + (self.gamma * next_q_values_t * (1-dones_t)) 

            q_values_t = self.critic(states_t, actions_t)

            self.critic_optimizer.zero_grad()
            critic_loss = self.criterion(q_values_t, targets_t)
            critic_loss.backward()
            self.critic_optimizer.step()

            proposed_actions_t = self.actor(states_t)
            proposed_q_values_t = self.critic(states_t, proposed_actions_t)

            self.actor_optimizer.zero_grad()
            actor_loss = - proposed_q_values_t.mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            utils.copy_model(self.actor, self.actor_target, tau=self.tau)
            utils.copy_model(self.critic, self.critic_target, tau=self.tau)

    def save_state(self):
        """ Save current state of agent """
        torch.save(self.actor, self.name + "_actor.model")
        torch.save(self.actor_target, self.name + "_actor_target.model")
        torch.save(self.critic, self.name + "_critic.model")
        torch.save(self.critic_target, self.name + "_critic_target.model")

        f_state = open(self.name + "_parameters.dat", "w")
        f_state.write("gamma = " + str(self.gamma) + "\n")
        f_state.write("tau = " + str(self.tau) + "\n")
        f_state.write("num_replays = " + str(self.num_replays))
        f_state.close()

    def load_state(self, enable_cuda = False):
        """ Load current state of agent """
        for line in open(self.name + "_parameters.dat", "r"):
            param, val = line.split(" = ")
            if "gamma" in param:
                self.gamma = float(val)
            elif "tau" in param:
                self.tau = float(val)
            elif "num_replays" in param:
                self.num_replays = int(val)
        
        self.actor = torch.load(self.name + "_actor.model")
        self.actor_target = torch.load(self.name + "_actor_target.model")
        
        self.critic = torch.load(self.name + "_critic.model")
        self.critic_target = torch.load(self.name + "_critic_target.model")

        if enable_cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.actor = self.actor.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)


    def run(self, env):
        """ Train agent in environment env

        Params
        ======
            env(Env): environment to train agent in
        """
        recent_scores = deque(maxlen=100)
        recent_max_scores = deque(maxlen=100)

        f = open("performance.log", "w")
        f.write("#Score\tAvg.Score\tMax_Score\tAvg.Max_Score\n")


        for e in range(self.num_episodes):
            scores = np.zeros(self.num_agents)
            states = env.reset()
            done = False
            
            self.noise.reset()

            timestep = 0

            while True:
                actions, states_t = self.act(states)

                next_states, rewards, dones, _ = env.step(actions)
                scores += rewards

                next_states_t = self.preprocess(next_states)

                for a in range(self.num_agents):
                    self.replay_buffer.append(states_t[a], actions[a], rewards[a], next_states_t[a], dones[a])

                self.learn()
                if self.positive_emphasis:
                    self.learn(positive=True)

                states = next_states

                timestep += 1

                if np.any(dones):
                    break


            if self.positive_emphasis:
                print("timestep", timestep)
                print(len(self.replay_buffer))
                # If this episode was more successful than previous ones, append trajectory to positive_replay_buffer
                if max(scores) > np.mean(recent_max_scores):
                    self.replay_buffer.copy_last_trajectory(self.positive_replay_buffer, self.num_agents * timestep)


            recent_scores.append(np.mean(scores))
            recent_max_scores.append(max(scores))
    
            print("Iteration %i: score: %f\taverage_score: %f max_score: %f\taverage_max_score: %f" % (e, np.mean(scores), np.mean(recent_scores), max(scores), np.mean(recent_max_scores)))

            f.write(str(np.mean(scores))+ "\t" + str(np.mean(recent_scores)) + "\t" + str(max(scores))+ "\t" + str(np.mean(recent_max_scores)) + "\n")

            f.flush()

            utils.copy_model(self.actor, self.actor_target, tau=self.tau)
            utils.copy_model(self.critic, self.critic_target, tau=self.tau)

            if e == self.save_after:
                self.save_state()

        f.close()


    def evaluate(self, env, num_episodes=100, delay=0.0):
        """ Evaluate agent performance in environment env

        Params
        ======
            env(Env): environment to train agent in
            num_episodes(int): number of episodes to run
            delay(float): time delay to make visualization slower
        """
        recent_scores = deque(maxlen=num_episodes)
        recent_losses = deque(maxlen=num_episodes)

        for e in range(num_episodes):
            scores = np.zeros(self.num_agents)
            states = env.reset()
            done = False

            while True:
                actions, states_t = self.act(states, add_noise=False)

                next_states, rewards, dones, _ = env.step(actions)
                scores += rewards

                next_states_t = self.preprocess(next_states)

                for a in range(self.num_agents):
                    self.replay_buffer.append(states_t[a], actions[a], rewards[a], next_states_t[a], dones[a])

                states = next_states

                if np.any(dones):
                    break

                time.sleep(delay)

            recent_scores.append(np.mean(scores))
            print("Iteration %i: score: %f\taverage_score: %f" % (e, np.mean(scores), np.mean(recent_scores)))


            time.sleep(10*delay)

        print("#"*20)
        print("Overall average: %f\tlast 100 episodes: %f" % (np.mean(recent_scores), np.mean(np.array(recent_scores)[-100:])))


