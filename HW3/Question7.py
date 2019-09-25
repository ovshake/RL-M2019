import numpy as np 
from random import random, randint
import matplotlib.pyplot as plt 

def check_outside(x,y):
	return x < 0 or x > 3 or y < 0 or y > 11 

def check_cliff(x,y):
	return x == 0 and y > 1 and y < 11

def check_terminal(x,y):
	return x == 0 and y == 11


def take_greedy(q_opt, cur_x, cur_y, nX, nY):
	rewards = []
	new_pos = []

	for i in range(4):
		new_x , new_y = cur_x + nX[i] ,  cur_y + nY[i]
		if check_outside(new_x, new_y):
			rewards.append(np.max(q_opt[cur_x][cur_y][:]))
			new_pos.append((cur_x,cur_y)) 
		elif check_cliff(new_x, new_y):
			rewards.append(np.max(q_opt[0][0][:])) 
			new_pos.append((0,0)) 
		else:
			rewards.append(np.max(q_opt[new_x][new_y][:]))
			new_pos.append((new_x,new_y)) 

	action = np.argmax(rewards) 
	pos = new_pos[action] 
	return np.max(rewards)  

def take_action(eps, q_opt,cur_x, cur_y, nX, nY):
	if random() < eps:
		action = randint(0,3) 
		new_x = cur_x + nX[action] 
		new_y = cur_y + nY[action] 
		if check_outside(new_x, new_y):
			return action, [cur_x,cur_y], -1 	
		elif check_cliff(new_x, new_y):
			return action, [0,0], -100 
		else:
			return action, [new_x, new_y], -1

	else:
		rewards = []
		new_pos = []

		for i in range(4):
			new_x , new_y = cur_x + nX[i] ,  cur_y + nY[i]
			if check_outside(new_x, new_y):
				rewards.append(-1)
				new_pos.append((cur_x,cur_y)) 
			elif check_cliff(new_x, new_y):
				rewards.append(-100) 
				new_pos.append((0,0)) 
			else:
				rewards.append(-1)
				new_pos.append((new_x,new_y)) 

		action = np.argmax(rewards) 
		pos = new_pos[action] 
		return action, pos , rewards[action] 

def SARSA(q_opt):
	alpha = 0.1
	eps = 0.1 
	nX = [0,-1,0,1] 
	nY = [1,0,-1,0]
	cliff_x = 0
	cliff_y = 11
	cur_x = 0 
	cur_y = 0 
	action, (_ , _) , _  = take_action(eps,q_opt, cur_x, cur_y, nX, nY)
	total_rewards = 0
	while not check_terminal(cur_x , cur_y):
		new_action, (new_x , new_y ), reward  = take_action(eps,q_opt, cur_x, cur_y, nX, nY)
		total_rewards += reward
		newer_action, (_, _), _ = take_action(eps,q_opt, new_x, new_y,nX, nY)
		q_opt[cur_x][cur_y][new_action] += alpha * (reward + q_opt[new_x][new_y][newer_action] - q_opt[cur_x][cur_y][new_action]) 
		cur_x, cur_y = new_x, new_y  

	return total_rewards 


def q_learning(q_opt):
	alpha = 0.1
	eps = 0.1 
	nX = [0,-1,0,1] 
	nY = [1,0,-1,0]
	cliff_x = 0
	cliff_y = 11
	cur_x = 0 
	cur_y = 0 
	total_rewards = 0 
	while not check_terminal(cur_x, cur_y):
		action, (new_x, new_y), reward = take_action(eps, q_opt, cur_x, cur_y, nX, nY) 
		total_rewards += reward
		greedy_reward = take_greedy(q_opt, new_x, new_y, nX, nY) 
		q_opt[cur_x][cur_y][action] += alpha * (reward + greedy_reward - q_opt[cur_x][cur_y][action])
		cur_x, cur_y = new_x, new_y

	return total_rewards




def montecarlo(num_episodes, method):
	q_opt = np.zeros((4,12, 4), dtype = int)
	rewards_epi_array = np.zeros(num_episodes) 
	for i in range(num_episodes):
		if i % 100 == 0:
			print(i)
		if method == 'sarsa':
			rewards_epi_array[i] =  SARSA(q_opt) 
		else:
			rewards_epi_array[i] = q_learning(q_opt) 

	return rewards_epi_array

	# plt.plot([i for i in range(num_episodes)], rewards_epi_array, label = method) 












if __name__ == '__main__':
	q_learning_avg = np.zeros((200,500))
	sarsa_average = np.zeros((200,500))
	for i in range(200):
		sarsa_average[i] = montecarlo(500, 'sarsa') 
		q_learning_avg[i] = montecarlo(500, 'Q-Learning')
	sarsa_average = np.mean(sarsa_average, axis = 0) 
	q_learning_avg = np.mean(q_learning_avg, axis = 0)
	plt.plot(range(500), sarsa_average , label = 'Sarsa') 
	plt.plot( range(500), q_learning_avg,label = 'Q-Learning')
	plt.xlabel('Episodes')
	plt.ylabel('Rewards') 
	plt.legend() 
	plt.savefig('Q-Learning vs Sarsa') 

