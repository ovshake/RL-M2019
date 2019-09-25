import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#help taken from this link: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter05/blackjack.py
HIT = 1
STICK = 0

def get_card():
	card = np.random.randint(1,14)
	card = min(card , 10) 
	return card 


def value_of_card(card):
	if card == 1:
		return 11 
	else: return card

player_strategy = [0 for i in range(22)]
for i in range(20):
	player_strategy[i] = HIT 
player_strategy[20] = player_strategy[21] = STICK 

dealer_strategy = [0 for i in range(22)]

for i in range(17):
	dealer_strategy[i] = HIT 

for i in range(17,22):
	dealer_strategy[i] = STICK 



def play_blackjack(player_strategy, dealer_strategy):
	player_sum = 0
	dealer_sum = 0
	player_usable_ace = False
	player_trajectory = []
	while player_sum < 12:
		card = get_card() 
		player_sum += value_of_card(card) 

		if player_sum >= 22:
			player_sum -= 10 
		else:
			player_usable_ace = True 


	dealer_card_1 = get_card() 
	dealer_card_2 = get_card()
	dealer_sum += value_of_card(dealer_card_1) + value_of_card(dealer_card_2)
	initial_state = [player_sum, player_usable_ace, dealer_card_1] 
	dealer_usable_ace = 1 in [dealer_card_1,dealer_card_2]
	if dealer_sum > 21:
		dealer_sum -= 10

	#player's turn

	while 1: 
		action = player_strategy[player_sum] 
		player_trajectory.append([player_sum, player_usable_ace, dealer_card_1])
		if action == STICK:
			break 
		new_card = get_card() 
		player_sum += value_of_card(new_card)
		player_ace_count = 1 if player_usable_ace else 0 

		if new_card == 1:
			player_ace_count += 1

		while player_sum > 21 and player_ace_count:
			player_sum -= 10 
			player_ace_count -= 1

		if player_sum > 21:
			return initial_state, -1, player_trajectory 

		player_usable_ace = (player_ace_count == 1)

	#dealer's turn 
	while 1:
		action = dealer_strategy[dealer_sum] 
		if action == STICK:
			break 
		new_card = get_card() 
		dealer_ace_count = 1 if dealer_usable_ace else 0 
		if new_card == 1:
			dealer_ace_count += 1

		dealer_sum += value_of_card(new_card) 

		while dealer_sum > 21 and dealer_ace_count:
			dealer_sum -= 10 
			dealer_ace_count -= 1

		if dealer_sum > 21:
			return initial_state, 1, player_trajectory 

	if player_sum > dealer_sum:
		return initial_state, 1, player_trajectory 
	elif player_sum == dealer_sum :
		return initial_state, 0, player_trajectory
	else:
		return initial_state, -1, player_trajectory 


def greedy_policy(player_sum, usable_ace, dealer_sum, states_values_action, states_values_action_count):
	usable_ace = int(usable_ace) 
	values = states_values_action[player_sum - 12, dealer_sum - 1, usable_ace, : ] / states_values_action_count[player_sum -12, dealer_sum - 1, usable_ace, : ]
	return np.random.choice([a for a,x in enumerate(values) if x == np.max(values)]) 


def play_blackjack_with_init_state(player_strategy, dealer_strategy, initial_state, init_action, states_values_action, states_values_action_count, greedy_strategy = False):
	player_sum = 0
	dealer_sum = 0
	player_trajectory = []
	player_sum , dealer_card_1, player_usable_ace = initial_state 
	#player's turn
	dealer_card_2 = get_card() 
	dealer_usable_ace = 1 in [dealer_card_1,dealer_card_2]
	while 1: 
		action = player_strategy[player_sum] if not greedy_strategy else greedy_policy(player_sum, player_usable_ace, dealer_card_1,states_values_action, states_values_action_count) 
		player_trajectory.append([[player_sum, player_usable_ace, dealer_card_1], action]) 
		if action == STICK:
			break 
		new_card = get_card() 
		player_sum += value_of_card(new_card)
		player_ace_count = 1 if player_usable_ace else 0 

		if new_card == 1:
			player_ace_count += 1

		while player_sum > 21 and player_ace_count:
			player_sum -= 10 
			player_ace_count -= 1

		if player_sum > 21:
			return initial_state, -1, player_trajectory 

		player_usable_ace = (player_ace_count == 1)

	#dealer's turn 
	while 1:
		action = dealer_strategy[dealer_sum] 
		if action == STICK:
			break 
		new_card = get_card() 
		dealer_ace_count = 1 if dealer_usable_ace else 0 
		if new_card == 1:
			dealer_ace_count += 1

		dealer_sum += value_of_card(new_card) 

		while dealer_sum > 21 and dealer_ace_count:
			dealer_sum -= 10 
			dealer_ace_count -= 1

		if dealer_sum > 21:
			return initial_state, 1, player_trajectory 

	if player_sum > dealer_sum:
		return initial_state, 1, player_trajectory 
	elif player_sum == dealer_sum :
		return initial_state, 0, player_trajectory
	else:
		return initial_state, -1, player_trajectory 

def monte_carlo_on_policy(num_episodes):
	states_usable_ace_present = np.zeros((10,10), dtype = int) 
	states_usable_ace_count = np.ones((10,10),dtype = int) 
	states_no_usable_ace_present = np.zeros((10,10), dtype = int) 
	states_no_usable_ace_count = np.ones((10,10),dtype = int) 
	for i in range(num_episodes):
		_, reward, player_trajectory = play_blackjack(player_strategy, dealer_strategy) 
		for (player_sum, usable_ace, dealer_card_1)  in player_trajectory:
			if usable_ace:
				states_usable_ace_present[player_sum - 12, dealer_card_1 - 1] += reward 
				states_usable_ace_count[player_sum - 12, dealer_card_1 - 1] += 1 
			else:
				states_no_usable_ace_present[player_sum - 12, dealer_card_1 - 1] += reward 
				states_no_usable_ace_count[player_sum - 12, dealer_card_1 - 1] += 1 


	return states_usable_ace_present / states_usable_ace_count, states_no_usable_ace_present / states_no_usable_ace_count 



def monte_carlo_ES(num_episodes):
	states_values_action = np.zeros((10,10,2,2))
	states_values_action_count = np.ones((10,10,2,2))




	for i in range(num_episodes):
		if i % 1000 == 0:
			print(i)
		initial_state = [np.random.choice(range(12, 22)), np.random.choice(range(1, 11)), bool(np.random.choice([0, 1]))]
		init_action = np.random.choice([HIT, STICK]) 
		if num_episodes == 0:
			_, reward, player_trajectory = play_blackjack_with_init_state(player_strategy, dealer_strategy, initial_state, init_action, states_values_action, states_values_action_count)
			for (player_sum, usable_ace, dealer_card) , action in player_trajectory:
				usable_ace = int(usable_ace) 
				states_values_action[player_sum - 12, dealer_card - 1, usable_ace] += reward 
				states_values_action_count[player_sum - 12, dealer_card - 1, usable_ace] += 1 
		else:
			_, reward, player_trajectory = play_blackjack_with_init_state(player_strategy, dealer_strategy, initial_state, init_action, states_values_action, states_values_action_count, greedy_strategy = True)
			for (player_sum, usable_ace, dealer_card) , action in player_trajectory:
				usable_ace = int(usable_ace) 
				states_values_action[player_sum - 12, dealer_card - 1, usable_ace] += reward 
				states_values_action_count[player_sum - 12, dealer_card - 1, usable_ace] += 1 

	return states_values_action / states_values_action_count
			
def play_black_jack_behaviour_policy(init_state):
	player_sum = 0
	player_trajectory = []
	player_usable_ace, player_sum, dealer_card1 = init_state 
	dealer_card2 = get_card() 
	dealer_sum = value_of_card(dealer_card2) + value_of_card(dealer_card1)
	dealer_usable_ace = False 
	if dealer_card1 == 1 or dealer_card2 == 1:
		dealer_usable_ace = True 

	if dealer_sum > 21:
		dealer_sum -= 10 
	
	while 1:
		if np.random.binomial(1,0.5) == 1:
			player_trajectory.append([[player_usable_ace, player_sum, dealer_card1], STICK])
			break
		player_trajectory.append([[player_usable_ace, player_sum, dealer_card1], HIT])
		new_card = get_card()
		player_ace_count = int(player_usable_ace)
		if new_card == 1:
			player_ace_count += 1
		player_sum += value_of_card(new_card) 

		while player_sum > 21 and player_ace_count > 0:
			# print(player_sum, player_ace_count)
			player_ace_count -= 10 
			player_ace_count -= 1

		if player_sum > 21:
			return init_state, -1, player_trajectory 

		player_usable_ace = (1 == player_ace_count)

	while 1: 
		action = dealer_strategy[dealer_sum]
		if action == STICK:
			break 

		new_card = get_card() 
		dealer_ace_count = int(dealer_usable_ace) 
		if new_card == 1:
			dealer_ace_count += 1

		dealer_sum += value_of_card(new_card) 
		while dealer_sum > 21 and dealer_ace_count:
			dealer_sum -= 10 
			dealer_ace_count -= 1

		if dealer_sum > 21:
			return init_state, 1, player_trajectory 
		dealer_usable_ace = (1 == dealer_ace_count) 

	if player_sum > dealer_sum:
		return init_state, 1, player_trajectory 

	elif dealer_sum == player_sum:
		return init_state, 0, player_trajectory 
	else:
		return init_state, -1, player_trajectory 


		


def monte_carlo_offpolicy(num_episodes):
	init_state = [True,13,2]
	rhos_array = [] 
	total_return_array = [] 
	for i in range(num_episodes):
		# print(i)
		_, reward, player_trajectory = play_black_jack_behaviour_policy(init_state) 
		num = 1.0 
		den = 1.0 
		for (player_usable_ace, player_sum, dealer_card1) , action in player_trajectory:
			if action == player_strategy[player_sum]:
				den *= (.5) 
			else:
				num = 0
				break 
		rho = num / den 
		rhos_array.append(rho) 
		total_return_array.append(reward)

	rhos_array = np.asarray(rhos_array) 
	total_return_array = np.asarray(total_return_array) 
	weighted_return_array = rhos_array * total_return_array 

	weighted_return_array = np.add.accumulate(weighted_return_array) 
	rhos_array = np.add.accumulate(rhos_array) 
	ordinary_sampling = weighted_return_array / np.arange(1, num_episodes + 1) 
	with np.errstate(divide='ignore',invalid='ignore'):
		weighted_sampling = np.where(rhos_array != 0, weighted_return_array / rhos_array, 0)

	return ordinary_sampling, weighted_sampling


def plot_figure_5_1():
	# fig , axes = plt.subplots(2, 2, figsize=(40, 30))
	# plt.subplots_adjust(wspace=0.1, hspace=0.2)
	# axes = axes.flatten()
	usable_ace_10000, no_usable_ace_10000 = monte_carlo_on_policy(10000) 
	usable_ace_500000, no_usable_ace_500000 = monte_carlo_on_policy(500000)
	values = [usable_ace_10000, no_usable_ace_10000,usable_ace_500000, no_usable_ace_500000]
	titles = ['Optimal value with usable Ace 10000','Optimal value without usable Ace 10000','Optimal value with usable Ace 500000', 'Optimal value without usable Ace 500000']
	player_range = np.arange(12,22)
	dealer_range = np.arange(1,11) 
	dealer_range , player_range = np.meshgrid(dealer_range, player_range)
	for value,title in zip(values,titles):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot_surface(dealer_range, player_range, value, cmap = cm.coolwarm) 
		ax.set_xlabel('dealer_sum')
		ax.set_ylabel('player_sum')
		ax.set_title(title) 
		plt.savefig(title + '.jpeg')



def plot_figure_5_2():
	values = monte_carlo_ES(500000) 
	state_usable_ace_vals = np.max(values[:,:,1,:], axis = -1)
	state_no_usable_ace_vals = np.max(values[:,:,0,:], axis = -1) 
	action_usable_ace = np.argmax(values[:,:,1,:], axis = -1)
	action_no_usable_ace = np.argmax(values[:,:,0,:], axis = -1)

	titles = ['Optimal policy with usable ace', 'optmimal policy without usable ace', 'Optimal Value with usable ace', 'Optimal Value Without Usable ace'] 
	values_array = [action_usable_ace, action_no_usable_ace, state_usable_ace_vals, state_no_usable_ace_vals] 
	player_range = np.arange(12,22)
	dealer_range = np.arange(1,11) 
	dealer_range , player_range = np.meshgrid(dealer_range, player_range)
	for value,title in zip(values_array,titles):
		fig = plt.figure()

		ax = fig.gca(projection='3d')

		ax.plot_surface(dealer_range, player_range, value, cmap = cm.coolwarm) 
		ax.set_xlabel('dealer_sum')
		ax.set_ylabel('player_sum')
		ax.set_title(title) 
		plt.savefig(title + '(5.2).jpeg')

def plot_figure_5_3():
	true_val = -0.27726
	num_episodes = 10000
	runs = 100 
	error_ordinary = np.zeros(num_episodes) 
	error_weighted = np.zeros(num_episodes) 
	for i in range(runs):
		print('Runs :',i)
		ordinary_sampling, weighted_sampling = monte_carlo_offpolicy(num_episodes) 
		error_ordinary += np.power(ordinary_sampling - true_val, 2)
		error_weighted += np.power(weighted_sampling - true_val, 2) 
	error_ordinary /= runs 
	error_weighted /= runs 
	plt.plot(error_ordinary, label='Ordinary Importance Sampling')
	plt.plot(error_weighted, label='Weighted Importance Sampling')
	plt.xlabel('Episodes (log scale)')
	plt.ylabel('Mean square error')
	plt.xscale('log')
	plt.legend()

	plt.savefig('Ordinary Importance Sampling Vs Weighted Importance Sampling.png')
	plt.close()

if __name__ == '__main__':
	plot_figure_5_3()
