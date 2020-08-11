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
