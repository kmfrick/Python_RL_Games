#!/usr/bin/env python
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent based on https://inst.eecs.berkeley.edu/~cs188/sp19/projects.html
        Attributes
          - self.epsilon (exploration prob)
          - self.alpha (learning rate)
          - self.discount (discount rate aka gamma)

        Methods
          - self.get_legal_actions(state) {state, hashable -> list of actions, each is hashable}
            which returns legal actions for a state
          - self.get_qvalue(state,action)
            which returns Q(state,action)
          - self.set_qvalue(state,action,value)
            which sets Q(state,action) := value
        """

        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """ Returns Q(state,action) """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """ Sets the Qvalue for [state,action] to the given value """
        self._qvalues[state][action] = value

    def get_value(self, state):
        """
        Compute agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action) over possible actions.
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0
        qvals = []
        for a in possible_actions:
          qvals.append(self.get_qvalue(state, a))
        return max(qvals)


    def update(self, state, action, reward, next_state):
        """
        Q-Value update here:
           Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """

        # agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        oldq = self.get_qvalue(state, action)
        newq = (1 - learning_rate) * oldq + learning_rate * (reward + gamma * self.get_value(next_state))
        self.set_qvalue(state, action, newq)

    def get_best_action(self, state):
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_actions = self.get_legal_actions(state)

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        idx = []
        qvals = []
        for a in possible_actions:
          qvals.append(self.get_qvalue(state, a))
          idx.append(a)

        return idx[np.argmax(qvals)]

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).
        """

        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        # If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        # agent parameters:
        epsilon = self.epsilon

        p = np.random.uniform(0, 1)
        if p < epsilon:
          return np.random.choice(possible_actions)
        else:
          return self.get_best_action(state)
    def print_qtable(self):
      for s in range(0, n_states):
        print("[", end="")
        for a in range(0, n_actions):
          print(p1.get_qvalue(s, a), end=", ")
        print("]")
      print("--------")


n_actions = 2 # Cooperate or defect
n_states = 2**2 # one-period memory, 2 actions for 2 players
epochs = 0
import random

class IteratedPrisonerDilemma:
    def __init__(self):
        self.states = [0, 1, 2, 3]
        self.reset()

    def reset(self):
        self.payoffs = [[(5, 5), (-5, 15)], [(15, -5), (0, 0)]]
        self.next_s = [[0, 1], [2, 3]]
        self.turn = 0
        self.p1action = -1
        self.p2action = -1
        self.s = random.randrange(0, n_states)
        if epochs > 1000:
            self.s = 3
        print("starting state:" + str(self.s))
        return 0

    def step(self, action):
        if self.turn == 0:
            self.turn = 1
            self.p1action = action
            self.p2action = -1
            return None
        elif self.turn == 1:
            self.turn = 0
            self.s = self.next_s[self.p1action][action]
            self.p2action = action
            return self.s, self.payoffs[self.p1action][self.p2action]
        else:
            raise ValueError

def play_and_train(env, p1, p2, t_max=10**4):
    """
    This function:
    - runs a full game, actions given by agent's e-greedy policy
    - trains agent using agent.update(...) whenever it is possible
    - returns total reward
    """
    total_reward = (0.0, 0.0)
    s = env.reset()

    for t in range(t_max):
        a1 = p1.get_action(s)
        env.step(a1)
        a2 = p2.get_action(s)
        next_s, (r1, r2) = env.step(a2)

        # train (update) agents for state s
        p1.update(s, a1, r1, next_s)
        p2.update(s, a2, r2, next_s)

        s = next_s
        total_reward = tuple(map(lambda i, j: i + j, total_reward, (r1, r2)))

    return total_reward




eps0 = 0.25
p1 = QLearningAgent(
    alpha=0.5, epsilon=eps0, discount=0.99,
    get_legal_actions=lambda s: range(n_actions))

p2 = QLearningAgent(
    alpha=0.5, epsilon=eps0, discount=0.99,
    get_legal_actions=lambda s: range(n_actions))
env = IteratedPrisonerDilemma()
rewards = []
epsilons = []

from IPython.display import clear_output

num_epochs = 4000

for i in range(num_epochs):
    reward = play_and_train(env, p1, p2)
    clear_output(wait=True)
    print("Epoch " + str(i))
    epochs +=1

    rewards.append(reward)
    print(reward)
    epsilons.append(p1.epsilon)
    print("Best actions for player 1: " + str([p1.get_best_action(s) for s in range(0, n_states)]))
    print("Best actions for player 2: " + str([p2.get_best_action(s) for s in range(0, n_states)]))


    p1.epsilon *= 0.995
    p2.epsilon *= 0.995


import matplotlib.pyplot as plt
import pandas as pd

# Moving average to smooth spikes 
# Usually, moving average is cheating, but this is computer science and not finance
# Drawing the same plot without moving average shows the same tendency
def moving_average(x, span=100):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values

for r in zip(*rewards):
    r = moving_average(r)
    plt.plot(range(0, len(rewards)), np.array(r))
plt.title('Rewards')
plt.xlabel('Period')
plt.ylabel('Reward')
plt.legend(['p1', 'p2'], loc='lower center')
plt.show()
plt.plot(np.array(epsilons))
plt.title('Exploration rate')
plt.xlabel('Period')
plt.ylabel('Epsilon')
plt.show()


