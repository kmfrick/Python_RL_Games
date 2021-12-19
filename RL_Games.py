import numpy as np
from os import system
import matplotlib.pyplot as plt
import pandas as pd

n_agents = 2
memory = 1		# 0 or 1 

alpha1 = 0.5
alpha2 = 0.5
eps0 = 0.25
gamma = 0.99

game_mode = "price_game_ext" # prisoner_dilemma, battle_sexes, stag_hunt, chicken, matching_pennies, rock_paper_scissor, optional_prisoner_dilemma, traveler_dilemma


class static_game():

	def __init__(self, game_mode):
		if game_mode == "prisoner_dilemma":
			self.payoffs = [[(5, 5), (-5, 15)],
							[(15, -5), (0, 0)]]		# cooperate or defect
			self.n_actions = 2
		elif game_mode == "battle_sexes":
			self.payoffs = [[(3, 2), (0, 0)],
							[(0, 0), (2, 3)]]
			self.n_actions = 2
		elif game_mode == "stag_hunt":
			self.payoffs = [[(4, 4), (1, 3)],
							[(3, 1), (2, 2)]]		# deer or rabbit
			self.n_actions = 2
		elif game_mode == "chicken":
			self.payoffs = [[(0, 0), (-1, 1)],
							[(1, -1), (-100, -100)]]	#
			self.n_actions = 2
		elif game_mode == "matching_pennies":
			self.payoffs = [[(1, -1), (-1, 1)],
							[(-1, 1), (1, -1)]]
			self.n_actions = 2
		elif game_mode == "rock_paper_scissor":
			self.payoffs = [[(0, 0), (-1, 1), (1, -1)],
							[(1, -1), (0, 0), (-1, 1)],
							[(-1, 1), (1, -1), (0, 0)]]	#rock, paper or scissor
			self.n_actions = 3
		elif game_mode == "optional_prisoner_dilemma":
			self.payoffs = [[(-1, -1), (-4, 0), (-2, -2)],
							[(0, -4), (-3, -3), (-2, -2)],
							[(-2, -2), (-2, -2), (-2, -2)]]	#cooperate, defect or abstain
			self.n_actions = 3
		elif game_mode == "price_game":
			self.payoffs = [[(0.204, 0.204), (0.297, 0.164), (0.352, 0.091)],
							[(0.164, 0.297), (0.304, 0.304), (0.427, 0.200)],
							[(0.091, 0.352), (0.2, 0.427), (0.336, 0.336)]]
			self.n_actions = 3
		elif game_mode == "price_game_ext":
			''' nash_price = 3, coop_price = 8'''
			self.payoffs = [[(0.1418729254244408, 0.1418729254244408), (0.16639886895570727, 0.15177529845848634), (0.18917799652823897, 0.14859562419604241), (0.2091176533071847, 0.1362828585204764), (0.22568652567836017, 0.11888809732305046), (0.23886917279074107, 0.09976733366536719), (0.24899979816831957, 0.08123897959982776), (0.2565792526838958, 0.06462450584256915), (0.26213711437995135, 0.050483021928854434), (0.2661527793329111, 0.03888052105616561)], [(0.15177529845848634, 0.16639886895570727), (0.18349688464144107, 0.18349688464144107), (0.2147612812233847, 0.18494417590559545), (0.24368018577005018, 0.17410852211456357), (0.2689009635133418, 0.1553010452036493), (0.2897922554455271, 0.13269797926441487), (0.3063713720504596, 0.10958797972102143), (0.31908657841783766, 0.08811168587639367), (0.32858526551412465, 0.06937678797126674), (0.3355426014971269, 0.053740043362949005)], [(0.14859562419604241, 0.18917799652823897), (0.18494417590559545, 0.2147612812233847), (0.22292666003062275, 0.22292666003062275), (0.2601391456834432, 0.2158344914914988), (0.2943648938454194, 0.19741646067026883), (0.32406211159232723, 0.17231422010250957), (0.3485550805006018, 0.14477763676572336), (0.36792429012215266, 0.11797738377102583), (0.38273865628485204, 0.09383908374420025), (0.39378252718000717, 0.07323557804431036)], [(0.1362828585204764, 0.2091176533071847), (0.17410852211456357, 0.24368018577005018), (0.2158344914914988, 0.2601391456834432), (0.2590823369772831, 0.2590823369772831), (0.3011070568500574, 0.24339023057362433), (0.339459720895628, 0.21755344871351578), (0.37250813541908656, 0.18648791316047264), (0.3996059716789632, 0.1544390535325386), (0.42093412491932514, 0.12438855628956397), (0.43718694519089657, 0.09799811412024012)], [(0.11888809732305046, 0.22568652567836017), (0.1553010452036493, 0.2689009635133418), (0.19741646067026883, 0.2943648938454194), (0.24339023057362433, 0.3011070568500574), (0.2905132313513475, 0.2905132313513475), (0.33579751947923475, 0.2662398119806819), (0.3766966051662045, 0.2333052035460377), (0.4116161686867145, 0.1968046608653119), (0.4400278916199324, 0.1608660097510875), (0.4622519926075542, 0.12818792478683394)], [(0.09976733366536719, 0.23886917279074107), (0.13269797926441487, 0.2897922554455271), (0.17231422010250957, 0.32406211159232723), (0.21755344871351578, 0.339459720895628), (0.2662398119806819, 0.33579751947923475), (0.3154039938197786, 0.3154039938197786), (0.3619601163515562, 0.2827469467340257), (0.40343907436245613, 0.24329063843043755), (0.4384345722316762, 0.20215913105993757), (0.46662782015638815, 0.16320874994084936)], [(0.08123897959982776, 0.24899979816831957), (0.10958797972102143, 0.3063713720504596), (0.14477763676572336, 0.3485550805006018), (0.18648791316047264, 0.37250813541908656), (0.2333052035460377, 0.3766966051662045), (0.2827469467340257, 0.3619601163515562), (0.3317099585465086, 0.3317099585465086), (0.3772078401555877, 0.2911997149131746), (0.4170518443764973, 0.2461735627575667), (0.45017271139583837, 0.20156483686451382)], [(0.06462450584256915, 0.2565792526838958), (0.08811168587639367, 0.31908657841783766), (0.11797738377102583, 0.36792429012215266), (0.1544390535325386, 0.3996059716789632), (0.1968046608653119, 0.4116161686867145), (0.24329063843043755, 0.40343907436245613), (0.2911997149131746, 0.3772078401555877), (0.3374904595088809, 0.3374904595088809), (0.3795116767776341, 0.2901791959706459), (0.415550554635243, 0.24101786647808449)], [(0.050483021928854434, 0.26213711437995135), (0.06937678797126674, 0.32858526551412465), (0.09383908374420025, 0.38273865628485204), (0.12438855628956397, 0.42093412491932514), (0.1608660097510875, 0.4400278916199324), (0.20215913105993757, 0.4384345722316762), (0.2461735627575667, 0.4170518443764973), (0.2901791959706459, 0.3795116767776341), (0.3314480703492684, 0.3314480703492684), (0.36789123250597827, 0.2790639127006489)], [(0.03888052105616561, 0.2661527793329111), (0.053740043362949005, 0.3355426014971269), (0.07323557804431036, 0.39378252718000717), (0.09799811412024012, 0.43718694519089657), (0.12818792478683394, 0.4622519926075542), (0.16320874994084936, 0.46662782015638815), (0.20156483686451382, 0.45017271139583837), (0.24101786647808449, 0.415550554635243), (0.2790639127006489, 0.36789123250597827), (0.313537818524607, 0.313537818524607)]]
			self.n_actions = 10

		self.n_states = self.n_actions ** (n_agents * memory)	# memory_lenght = 1
		self.states = list(range(0, self.n_states))
		self.next_state = np.array_split(self.states, self.n_actions)
		self.initial_state = 0

	def reset_initial_state(self):
		self.initial_state = np.random.choice(self.states)
		return self.initial_state

	def get_payoffs(self, action1, action2):
		return self.payoffs[action1][action2]

	def get_next_state(self, action1, action2):
		if memory > 0:
			return self.next_state[action1][action2]
		else:
			return 0

class agent():

	def __init__(self, game, alpha, epsilon, gamma):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		n_states = game.n_states
		n_actions = game.n_actions
		self.Q = np.zeros([n_states, n_actions], dtype=np.float64) 
		self.strategy_set = list(range(0,n_actions))

	def get_best_action(self, state):
		"""
		Compute the best action in a given state according to the current q-values
		"""
		return np.argmax(self.Q[state])

	def get_action(self, state):
		"""
		Computes current action under e-greedy exploration
		"""
		p = np.random.uniform(0, 1)
		if p < self.epsilon:
			#exploration
			return np.random.choice(self.strategy_set)
		else:
			#exploitation (greedy action)
			return self.get_best_action(state)

	def update_q(self, state, action, reward, next_state):
		"""
		Q-learning update: 	Q(s,a) <- (1 - alpha) * Q(s,a) + alpha * (R + gamma * V(s')) 
							where V(s') = max_{a in A} Q(s',a) is the continuation value
		"""
		learning_rate = self.alpha
		discount_rate = self.gamma

		continuation_value = self.Q[next_state][self.get_best_action(next_state)]
		self.Q[state][action] = (1 - learning_rate) * self.Q[state][action] + learning_rate * (reward + discount_rate * continuation_value)


def training(game, player1, player2, episode_max = 10**4):

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


def show_epoch_outcome(i, reward, freq_states, freq_actions1, freq_actions2, epsilon, game_mode, payoffs):
	system("clear")		# unix based systems only
	print("Game: \t " + game_mode)
	print("Payoffs: " + str(payoffs), end="\n\n")
	print("Period " + str(i))
	print("Limit strategies \t" + str(best_action_1) + " , " + str(best_action_2))
	print("Total reward \t\t" + str(reward), end="\n\n")
	print("Frequency states \t" + str(np.around(freq_states,3)))
	print("Freq. player1's actions " + str(np.around(freq_actions1,3)))
	print("Freq. player2's actions " + str(np.around(freq_actions2,3)))
	# within brakets: probability of at least one agent exploring on an episode
	print("Epsilon \t\t " + str(np.around(epsilon,10)) + "\t ( "+str(np.around((epsilon**2 + 2*(epsilon)*(1-epsilon))*100,5)) + "% )", end="\n\n")

	#print(np.around(player1.Q), end = "\n\n")
	#print(np.around(player2.Q), end = "\n\n")


# initialize game and players
game = static_game(game_mode)
player1 = agent(game, alpha1, eps0, gamma)
player2 = agent(game, alpha2, eps0, gamma)

# convergence rules
convergence_target = 500
same_best_action_t = 0

# init best actions vectors (full of -1)
best_action_1 = np.full(game.n_states, -1)
best_action_2 = np.full(game.n_states, -1)

rewards = []
epsilons = []


# loop over epochs
for epoch in range(10**5):
	# play, learn, and get rewards (+ stats)
	freq_actions1, freq_actions2, freq_states, reward = training(game, player1, player2)

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
	show_epoch_outcome(epoch, reward, freq_states, freq_actions1, freq_actions2, player1.epsilon, game_mode, game.payoffs)
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
