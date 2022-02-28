#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from typing import List
from numba import njit

# Pure-strategy Nash equilibrium finder on matrix games
# Originally by Allison Oldham Luedtke
# Department of Economics, Saint Michael’s College, Colchester, Vermont, United States
# aluedtke@smcvt.edu
@njit
def find_pure_nash(M: np.ndarray):
    m = M.shape[0]
    n = M.shape[1]
    label = np.zeros((m, n))
    for row in range(m):
        for col in range(n):
            # check row payoffs
            for r in range(m):
                if label[row][col] == 0:
                    if M[r][col][0] > M[row][col][0]:
                        # this is not a Nash equilibrium
                        label[row][col] = 2
                    # only bother checking if it's still unset
            if label[row][col] == 0:
                for c in range(n):
                    if label[row][col] == 0:
                        if M[row][c][1] > M[row][col][1]:
                            # not a Nash equilibrium
                            label[row][col] = 2
            # if we made it all this way without setting to 2, it's a Nash equilibrium
            if label[row][col] == 0:
                label[row][col] = 1
    eq = []
    for the_row in range(m):
        for the_col in range(n):
            if label[the_row][the_col] == 1:
                eq.append((the_row, the_col))
    return list(map(tuple, eq))


@njit
def init_q_matrix(
    n_states: int, n_actions: int, payoffs: np.ndarray, discount: float, i: int
) -> np.ndarray:
    if i > 1:
        print("More than two agents not implemented yet")
        return None
    Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            if i == 0:
                Q[s][a] = np.mean(payoffs[a, :, 0]) / (1 - discount)
            elif i == 1:
                Q[s][a] = np.mean(payoffs[:, a, 1]) / (1 - discount)
    return Q


@njit
def get_action_ql(Q: np.ndarray, state: int, er_decay: float, t: int):
    """
    Computes current action under e-greedy exploration
    """
    p = np.random.uniform(0, 1)
    er = np.exp(-er_decay * t)
    if p < er:
        # exploration
        return np.random.randint(Q.shape[1])  # Q is [states, actions]
    else:
        # exploitation (greedy action)
        return np.argmax(Q[state])


@njit
def update_q(
    Q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    lr: float,
    discount: float,
):
    """
    Q-learning update:     Q(s,a) <- (1 - lr) * Q(s,a) + lr * (R + discount * V(s'))
                        where V(s') = max_{a in A} Q(s',a) is the continuation value
    """
    continuation_value = Q[next_state][np.argmax(Q[next_state])]
    Q[state][action] = (1 - lr) * Q[state][action] + lr * (
        reward + discount * continuation_value
    )


@njit
def play_and_train(
    payoffs: np.ndarray,
    n_states: int,
    n_actions: int,
    next_state_for_actions: np.ndarray,
    df: float,
    it_max: int,
    er_decay: float,
    ql_lr: float,
    rep_buf_size: int,
    batch_size: int,
):
    state = np.random.randint(n_states)
    reward_from_cur_expert1 = 0.0
    reward_from_cur_expert2 = 0.0
    q1 = init_q_matrix(n_states, n_actions, payoffs, df, 0)
    q2 = init_q_matrix(n_states, n_actions, payoffs, df, 1)
    rewards = np.zeros((it_max, 2))
    # loop over periods
    for t in range(1, it_max):
        # State is read by get_action()
        action1 = get_action_ql(q1, state, er_decay, t)
        action2 = get_action_ql(q2, state, er_decay, t)

        # collect instantaneous reward and observe next period state
        reward1, reward2 = payoffs[action1, action2]
        rewards[t - 1, 0] = reward1
        rewards[t - 1, 1] = reward2
        next_state = next_state_for_actions[action1, action2]
        update_q(q1, state, action1, reward1, next_state, ql_lr, df)
        update_q(q2, state, action2, reward2, next_state, ql_lr, df)

        # set state to next state and iterate
        state = next_state
    return rewards


def main():
    # Learning rate and exploration rate decay have to be gridsearched for every game
    ql_lr = 5e-2
    er_decay = 7e-4
    memory = 1
    it_max = int(5e4)
    trial_max = 100
    df = 0.95
    rep_buf_size = int(50)
    batch_size = 4
    payoffs_for_game = {
        "prisoner_dilemma": [[(3, 3), (0, 5)], [(5, 0), (1, 1)]],
        "battle_sexes": [[(3, 2), (0, 0)], [(0, 0), (2, 3)]],
        "stag_hunt": [[(4, 4), (1, 3)], [(3, 1), (2, 2)]],
        "chicken": [[(0, 0), (-1, 1)], [(1, -1), (-100, -100)]],
        "matching_pennies": [[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]],
        "rock_paper_scissor": [
            [(0, 0), (-1, 1), (1, -1)],
            [(1, -1), (0, 0), (-1, 1)],
            [(-1, 1), (1, -1), (0, 0)],
        ],
        "optional_prisoner_dilemma": [
            [(-1, -1), (-4, 0), (-2, -2)],
            [(0, -4), (-3, -3), (-2, -2)],
            [(-2, -2), (-2, -2), (-2, -2)],
        ],
    }
    game = "prisoner_dilemma"
    payoffs = np.array(payoffs_for_game[game])
    n_actions = payoffs.shape[
        0
    ]  # Assumption: all agents have the same number of actions
    n_agents = len(payoffs.shape) - 1
    n_states = n_actions ** (n_agents * memory)
    states = np.arange(0, n_states)
    next_state_for_actions = states.reshape((n_actions, n_actions))
    nash_actions = find_pure_nash(payoffs)
    nash_payoff = np.max(
        np.array([payoffs[n][0] for n in nash_actions])
    )  # Assumption: symmetric matrix
    payoffs1 = payoffs[:, :, 0]
    payoffs2 = payoffs[:, :, 1]
    sucker_payoff = np.min(
        payoffs
    )  # Assumption: the worst payoff is obtained when being taken advantage of
    coop_payoff = 0
    for i in range(n_actions):
        for j in range(n_actions):
            if payoffs1[i, j] == payoffs2[i, j]:
                coop_payoff = max(coop_payoff, payoffs1[i, j])

    coop_actions = [
        (i, j)
        for i in range(n_actions)
        for j in range(n_actions)
        if coop_payoff in payoffs[i, j]
    ]
    coop_states = [next_state_for_actions[a] for a in coop_actions]
    coop_actions = np.array(coop_actions)
    print(nash_actions)
    print(nash_payoff)
    print(coop_actions)
    print(coop_payoff)
    reward_series = np.zeros((trial_max, it_max, n_agents))
    for trial in tqdm(range(trial_max)):
        rewards = play_and_train(
            payoffs,
            n_states,
            n_actions,
            next_state_for_actions,
            df,
            it_max,
            er_decay,
            ql_lr,
            rep_buf_size,
            batch_size,
        )
        reward_series[trial, :] = rewards

    reward_series = np.mean(reward_series, axis=0)
    gain_series = (reward_series - nash_payoff) / (coop_payoff - nash_payoff)

    mov_avg_int = int(1e2)

    for r in zip(*gain_series):
        # Moving average to smooth spikes
        r = pd.DataFrame({"r": np.asarray(r)}).r.ewm(span=mov_avg_int).mean().values
        plt.plot(range(0, len(gain_series)), np.array(r))
    plt.title("Profit gains")
    plt.xlabel("Period")
    plt.ylabel("Profit gain")
    plt.axhline(np.mean(gain_series))
    plt.legend(["p1", "p2"], loc="lower center")
    plt.show()

    plt.plot(np.exp(-er_decay * np.arange(it_max)))
    plt.show()


if __name__ == "__main__":
    main()
