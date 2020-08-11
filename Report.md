### Description of the Implemented Approach

The approach implemented in this repository is called **Deep Deterministic Policy Gradient** (DDPG) and is based on *reinforcement learning*, i.e. it is a machine learning approach in which an agent tries to improve his performance by interacting with the environment. In every step *t* of an episode, the agent chooses an action as described before *A<sub>t</sub>* depending on the state *S<sub>t</sub>* he is in and observes the next state as well as a response in the form of a reward *R<sub>t</sub>* which is a real number in general. In this implementation, the algorithm is *value-based*, which means that the agent chooses the action by consulting an *action-value function*    
<p align="center"> <img src="https://latex.codecogs.com/svg.latex?&space;q_\pi(s,a)" /></p>

which is given as the expected return *G* when taking action *a* in state *s* and subsequently following policy <img src="https://latex.codecogs.com/svg.latex?\pi" />:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?q_%5Cpi%28s%2Ca%29%3D%5Cleft%3CG_t%7CS_t%3Ds%2CA_t%3Da%5Cright%3E_%5Cpi%3D%5Cleft%3C%5Cleft.%5Csum_%7Bk%3D0%7D%5E%5Cinfty%5Cgamma%5EkR_%7Bt&plus;k&plus;1%7D%5Cright%7CS_t%3Ds%2CA_t%3Da%5Cright%3E_%5Cpi" /> </p>

In this equation <img src="https://latex.codecogs.com/svg.latex?&space;0\leq\gamma<1" /> is a discounting factor that describes how valuable future rewards are compared to present ones and ensures that the expected return *G* is finite as long as the reward sequence *{ R<sub>k</sub> }* is bounded.

To learn this action-value function the agent makes an (typically very poor) initial guess for the action-value function and updates it according to an *update rule*. The update rule chosen here is a *1-step Temporal-Difference (TD) Learning* update rule, which for a sequence of *(state, reward, action, next state, next action)* *(S<sub>t</sub>, R<sub>t</sub>, A<sub>t</sub>, S<sub>t+1</sub>, A<sub>t+1</sub>)* reads

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?q_\pi(S_t,A_t)=q_\pi(S_t,A_t)+\alpha\left[R_t+\gamma\,q_\pi(S_{t+1},A_{t+1})-q_{\pi}(S_{t},A_{t})\right]\" /></p>

Here, <img src="https://latex.codecogs.com/svg.latex?\alpha" /> is the so called *learning-rate*, which is typically chosen quite small to improve the convergence of the updates by decreasing the fluctuations. In principle, this action-value function can be used to calculate the best action to take given state *S<sub>t</sub>* by calculating its *argmax*. However, in some cases, this is not feasible as the action-space is too large to calculate the *argmax* of the action-value function. This is especially the case for continuous actions, as in this case the action-space is infinitely large. In DDPG this is solved as follows: There are two neural networks involved, one which approximates the action-value function, called the *critic* and one network which tries to approximate the *argmax* of the action-value function, called the *actor*. The actor can be optimized by applying *gradient ascent* to the action value function:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\theta=\theta+\alpha_\theta\nabla_{\theta}E\left[q_\pi(S_t,\mu_\theta(S_t))\right]" /></p>

Here, <img src="https://latex.codecogs.com/svg.latex?\theta" /> corresponds to the actor's network weights, while  <img src="https://latex.codecogs.com/svg.latex?\mu_\theta(S_t)" /> is the action proposed by the actor given the state *S<sub>t</sub>* and the network weights <img src="https://latex.codecogs.com/svg.latex?\theta" />. The expectation in the expression above is taken with respect to the observations sampled from previous observations and <img src="https://latex.codecogs.com/svg.latex?\alpha_\theta" /> is the actor's learning rate.

Now the procedure for both of the agent is as follows:
1. The agent observes a state
2. The agent's actor chooses an action by approximating the *argmax* of the action-value function <img src="https://latex.codecogs.com/svg.latex?A_t=\mu_\theta(S_t)\approx\;argmax_a\,q_\pi(S_t,a)\" />
3. The agent observes the reward, the next state and whether this next state is terminal.
4. During training, he updates the critic's network weights <img src="https://latex.codecogs.com/svg.latex?\phi" /> using a *mean-squared Bellmann error* (MSBE):

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?L(\phi)=E\left[\left(q_\phi(S_t,A_t)-\left(R+\gamma\,q_\phi(S_{t+1},\mu_\theta(S_t)\right)\right)^2\right]" /></p>

where <img src="https://latex.codecogs.com/svg.latex?q_\phi(S_t,A_t)" /> is the critics estimate of <img src="https://latex.codecogs.com/svg.latex?q_\pi(S_t,A_t)" />

5. During training, he updates the actor's network weights <img src="https://latex.codecogs.com/svg.latex?\theta" /> as follows:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\theta=\theta+\alpha_\theta\nabla_{\theta}E\left[q_\pi(S_t,\mu_\theta(S_t))\right]" /></p>

To facilitate exploration, the actions proposed by the actor were altered by adding some *Ornstein-Uhlenbeck-noise* when interacting with the environment. To ensure, that the actions still lie in the allowed range, the actions are clipped to the corresponding range after the addition of the noise.

An improvement to the algorithm was the usage of *replay buffers*: Replay buffers are storages for sequences observed by the agent while interacting with the environment. The memories in the replay buffer can be used to train the agent while not actually interacting with the environment by reusing previous observations. This leads to a more efficient usage of experiences, in turn making learning more efficient. Besides that, it typically leads to better generalization, as the agent is trained on potentially old memories, so that it does not forget about previous experiences and so that it is subject to a larger variety of different situations. Replay buffers can be considered a very simple "model of the environment" in that they assume that memories from the past are representative for the underlying dynamics of the environment.

To make training more stable, *fixed Q-targets* were used. In this technique, the agent uses two neural networks of the same architecture, where one is network not trained via gradient descent but whose weigths <img src="https://latex.codecogs.com/svg.latex?\omega" /> are updated using soft updates:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\omega=\tau\omega^{\prime}+(1-\tau)\omega" /></p>

here, <img src="https://latex.codecogs.com/svg.latex?\omega^{\prime}" /> are the weights of the neural network that is trained using some form of gradient descent.


### Network Architecture and Hyperparameters

The neural networks used here were simple *dense networks*, i.e. they consist of fully connected layers only. The architecture for the actor was as follows:

- Input layer of size `3x8` (corresponding to the 3 stacks of 8-dimensional states)
- Hidden layer with `64` neurons and `ReLU`-activation
- Hidden layer with `16` neurons and `ReLU`-activation
- Output layer with `2` neurons (corresponding to the 2-dimensional actions) with `tanh` activation function

The architecture for the critic was as follows:

- Input layer of size `3x8` (corresponding to the 3 stacks of 8-dimensional states)
- Hidden layer with `64` neurons and `ReLU`-activation
- Concatenation of 2-dimensional actions, hence making this layer `26`-dimensional
- Hidden layer with `16` neurons and `ReLU`-activation
- Output layer with `1` neuron (corresponding to the action-value) without activation function, i.e. linearly activated.

The hyperparameters used are given in the table below:

| Hyperparameter   |      Value      |
|----------|:-------------:|
| Q-target parameter <img src="https://latex.codecogs.com/svg.latex?\tau" /> |  0.1  |
| Discount factor <img src="https://latex.codecogs.com/svg.latex?\gamma" /> |    0.99   |
| Batchsize | 2^7 |
| Size of replay buffer | 2^20 |
| Number of replays per learning phase | 2 |
| Actor learning rate <img src="https://latex.codecogs.com/svg.latex?\alpha_\theta" /> | 0.0001 |
| Critic learning rate <img src="https://latex.codecogs.com/svg.latex?\alpha_\phi" /> | 0.0001 |
| Ornstein-Uhlenbeck mean reversion level <img src="https://latex.codecogs.com/svg.latex?\mu_{OU}" /> | 0 |
| Ornstein-Uhlenbeck mean reversion rate <img src="https://latex.codecogs.com/svg.latex?\theta_{OU}" /> | 0.15 |
| Ornstein-Uhlenbeck diffusion constant <img src="https://latex.codecogs.com/svg.latex?\sigma_{OU}" /> | 0.2 |



### Results

The reinforcement agents as described above reach the required average score (averaged over the last 100 episodes) of **`+0.5`** after about 4100 episodes. At their peak performance they achieve an average score of more than **`+0.93`** in training mode, while in evaluation mode they reach an astonishing average score of more than **`+2.5`**. In evaluation mode the ball was dropped only three times in 100 matches, with all the other matches reaching the timeout limit, which implies that the performance is close to optimal. Hence, the agents learned to cooperate very well and master this task of playing tennis with flying colors.

![results](https://github.com/fberressem/Tennis/blob/master/Results.png)

In general, the agent performs quite well. However, in the following there are some suggestions on how performance and especially the training speed may still be improved.



### Future Improvements

There are some improvements that may be implemented in the future:

1. To make training more stable, one might change the fixed Q-targets part to *double Q-Learning* such that the choice of actions while interacting with the environment is done using either one of the neural networks with the update rules being applied accordingly. In this case, there would be no dedicated *target network* anymore, while training should still be improved, as the networks would still be regularizing each other.

2. One might modify the rewards the agent sees to steer its behavior, e.g. by punishing giving an reward for small-valued actions to avoid hectic behaviour. However, this has to be done with great care as to avoid the agent learning a wrong behavior by relying on those additional rewards.

3. One could add prioritization in the replay buffer which could take into account how successful (or unsuccessful) the episodes they stem from were. This might improve stability of learning as learning about very good or very bad actions would be emphasized. In this implementation I tried to add an additional replay buffer for positive examples, such that the (in the beginning rarely occuring) positive experiences are utilized more efficiently. However, this did not significantly improve the performance in the conducted tests, which may be due to an additionally introduced bias by reusing these memories more often than others.

4. The final experiences from all the episodes ending due to the time limit could be disregarded when learning, as they are not representative for the value of the state. That is, the agent does not know about whether the next state is a terminal one or not when calculating its expected return, so that when training on this memory, it skews the value of the state, making training less stable. Hence, the performance might be improved by disregarding all next-to-terminal states in the training phase.

5. It might be beneficial to use **Trust Region Policy Optimization** (TRPO) rather than gradient descent based methods, as they guarantee policy improvement in every step and convergence to the optimum under some (often met) conditions. This might lead to a significant improvement, as the application of gradient descent is quite unstable with two agents interacting with each other because for both agents, the environment seems to be very unstable.

5. Better architectures might be found using grid-searching or more sophisticated methods, like evolutionary algorithms. Given that the agent already performs quite well, this might be fruitful thing to, as in principle the agent is working and on only wants to step training up a notch. 

