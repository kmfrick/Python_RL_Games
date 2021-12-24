import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time

from tqdm import tqdm

class StaticGame():

    payoffs_for_game = {'prisoner_dilemma': [[(5, 5), (-5, 15)], [(15, -5), (0, 0)]],
                        'battle_sexes': [[(3, 2), (0, 0)], [(0, 0), (2, 3)]],
                        'stag_hunt': [[(4, 4), (1, 3)], [(3, 1), (2, 2)]],
                        'chicken': [[(0, 0), (-1, 1)], [(1, -1), (-100, -100)]],
                        'matching_pennies': [[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]],
                        'rock_paper_scissor': [[(0, 0), (-1, 1), (1, -1)], [(1, -1), (0, 0), (-1, 1)], [(-1, 1), (1, -1), (0, 0)]],
                        'optional_prisoner_dilemma': [[(-1, -1), (-4, 0), (-2, -2)], [(0, -4), (-3, -3), (-2, -2)], [(-2, -2), (-2, -2), (-2, -2)]],
                        'price_game': [[(0.204, 0.204), (0.297, 0.164), (0.352, 0.091)], [(0.164, 0.297), (0.304, 0.304), (0.427, 0.2)], [(0.091, 0.352), (0.2, 0.427), (0.336, 0.336)]]}
    def __init__(self, game, n_agents, memory):
        self.game_mode = game
        self.payoffs = self.payoffs_for_game[game]
        self.n_actions = len(self.payoffs)
        self.memory = memory
        self.rng = np.random.default_rng(int(time.time()))
        self.n_states = self.n_actions ** (n_agents * memory)    # memory_length = 1
        self.states = list(range(0, self.n_states))
        self.next_state = np.array_split(self.states, self.n_actions)
        self.initial_state = 0

    def reset_initial_state(self):
        self.initial_state = self.rng.choice(self.states)
        return self.initial_state

    def get_payoffs(self, action1, action2):
        return self.payoffs[action1][action2]

    def get_next_state(self, action1, action2):
        if self.memory > 0:
            return self.next_state[action1][action2]
        else:
            return 0

class Agent():

    def __init__(self, game, alpha, epsilon, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.game = game
        self.Q = np.zeros([game.n_states, game.n_actions])
        self.strategy_set = list(range(0,game.n_actions))

    def get_best_action(self, state):
        '''
        Compute the best action in a given state according to the current q-values
        '''
        return np.argmax(self.Q[state])

    def get_action(self, state):
        '''
        Computes current action under e-greedy exploration
        '''
        p = self.game.rng.uniform(0, 1)
        if p < self.epsilon:
            #exploration
            return self.game.rng.choice(self.strategy_set)
        else:
            #exploitation (greedy action)
            return self.get_best_action(state)

    def update_q(self, state, action, reward, next_state):
        '''
        Q-learning update:     Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * (R + gamma * V(s'))
                            where V(s') = max_{a in A} Q(s',a) is the continuation value
        '''
        learning_rate = self.alpha
        discount_rate = self.gamma

        continuation_value = self.Q[next_state][self.get_best_action(next_state)]
        self.Q[state][action] = (1 - learning_rate) * self.Q[state][action] + learning_rate * (reward + discount_rate * continuation_value)


def train(game, player1, player2, episode_max = 10**4):

    # reset initial state at the beginning of a new epoch
    state = game.reset_initial_state()
    # init arrays
    total_reward1 = 0.0
    total_reward2 = 0.0
    freq_actions1 = np.zeros(game.n_actions)
    freq_actions2 = np.zeros(game.n_actions)
    freq_states = np.zeros(game.n_states)

    # loop over episodes
    for episode in range(episode_max):
        # get epsilon-greedy action given current state
        action1 = player1.get_action(state)
        action2 = player2.get_action(state)

        # collect instantaneous reward and observe next period state
        reward1, reward2 = game.get_payoffs(action1, action2)
        next_state = game.get_next_state(action1, action2)

        # update q value for the current state according to learning rate, reward and continuation value
        player1.update_q(state, action1, reward1, next_state)
        player2.update_q(state, action2, reward2, next_state)

        # set state to next state and iterate
        state = next_state

        # save rewards and basic stats
        total_reward1 += reward1
        total_reward2 += reward2
        freq_actions1[action1] += 1
        freq_actions2[action2] += 1
        freq_states[state] += 1

    return np.divide(freq_actions1,episode_max), np.divide(freq_actions2,episode_max), np.divide(freq_states,episode_max), (total_reward1, total_reward2)


def show_epoch_outcome(i, reward, best_action_1, best_action_2, freq_states, freq_actions1, freq_actions2, epsilon, game, payoffs, players):
    print('Game: \t ' + game.game_mode)
    print('Payoffs: ' + str(payoffs), end='\n\n')
    print('Period ' + str(i))
    print('Limit strategies \t' + str(best_action_1) + ' , ' + str(best_action_2))
    print('Total reward \t\t' + str(reward), end='\n\n')
    print('Frequency states \t' + str(np.around(freq_states,3)))
    print('Freq. player1''s actions ' + str(np.around(freq_actions1,3)))
    print('Freq. player2''s actions ' + str(np.around(freq_actions2,3)))
    # within brackets: probability of at least one agent exploring on an episode
    print('Epsilon \t\t ' + str(np.around(epsilon,10)) + '\t ( '+str(np.around((epsilon**2 + 2*(epsilon)*(1-epsilon))*100,5)) + '% )', end='\n\n')

    for player in players:
        for s in range(game.n_states):
            print("[", end="")
            for a in range(game.n_actions):
                print(player.Q[s, a], end=", ")
            print("]")


def main():

    n_agents = 2
    memory = 1             # 0 or 1

    alpha1 = 0.5
    alpha2 = 0.5
    eps0 = 0.25
    gamma = 0.99
    # prisoner_dilemma, battle_sexes, stag_hunt, chicken, matching_pennies, rock_paper_scissor, optional_prisoner_dilemma, traveler_dilemma
    game_mode = "prisoner_dilemma"

    # initialize game and players
    game = StaticGame(game_mode, n_agents, memory)
    player1 = Agent(game, alpha1, eps0, gamma)
    player2 = Agent(game, alpha2, eps0, gamma)

    # convergence rules
    convergence_target = 500
    same_best_action_t = 0

    # init best actions vectors (full of -1)
    best_action_1 = np.full(game.n_states, -1)
    best_action_2 = np.full(game.n_states, -1)

    rewards = []
    epsilons = []


    # loop over epochs
    for epoch in tqdm(range(10**5)):
        # play, learn, and get rewards (+ stats)
        freq_actions1, freq_actions2, freq_states, reward = train(game, player1, player2)

        # save past epoch best actions
        best_action_1_old = np.copy(best_action_1)
        best_action_2_old = np.copy(best_action_2)

        # get new best actions
        for s in game.states:
            best_action_1[s] = player1.get_best_action(s)
            best_action_2[s] = player2.get_best_action(s)

        # assess convergence
        if np.array_equiv(best_action_1,best_action_1_old) and np.array_equiv(best_action_2,best_action_2_old):
            same_best_action_t += 1
        else:
            same_best_action_t = 0
        if same_best_action_t == convergence_target:
            break

        # output
        show_epoch_outcome(epoch, reward, best_action_1, best_action_2, freq_states, freq_actions1, freq_actions2, player1.epsilon, game, game.payoffs, [player1, player2])
        rewards.append(reward)
        epsilons.append(player1.epsilon)

        # epsilon is decreasing
        player1.epsilon *= 0.99
        player2.epsilon *= 0.99



    for r in zip(*rewards):
        # Moving average to smooth spikes
        r = pd.DataFrame({'r': np.asarray(r)}).r.ewm(span=50).mean().values
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

if __name__ == '__main__':
    main()
