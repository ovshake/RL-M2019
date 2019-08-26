import numpy as np 
import matplotlib.pyplot as plt 
import random 
import scipy as sp 
from math import sqrt , log
# 10 armed test bed with non stationary reward distribution 
# 2000 such experiments each for 1000 such steps 
np.random.seed(0) 
step_size = 1
total_steps = 5000
def stationary_ten_armed_testbed(eps, q_estimates , num_steps = total_steps, initial_reward = 0, nonstationary = False):
	# optimal_move = np.argmax(q_estimates) 
	q_estimates = np.asarray(q_estimates)
	moves_count = np.zeros(10)
	rewards_till_now = [] 
	reward = 0 
	avg_reward = np.zeros(10) + initial_reward
	per_optimal_action = [] 
	optimal_move_counter = 0
	def get_best_move(array):
		return np.argmax(array) 
	
	for step in range(1,num_steps+1):
		if nonstationary:
			q_estimates += np.random.normal(0,0.01,10)
			# print(q_estimates)

		epsilon = random.random() 
		if epsilon > eps: 
			best_move = get_best_move(avg_reward) 
			moves_count[best_move] += 1
			if best_move == np.argmax(q_estimates) :
				optimal_move_counter += 1

			noise = np.random.normal(0,1)
			reward = q_estimates[best_move] + noise 
			avg_reward[best_move] += (reward - avg_reward[best_move]) / moves_count[best_move]
		else:
			move = random.randint(0,9) 
			if move == np.argmax(q_estimates) :
				optimal_move_counter += 1
			moves_count[move] += 1
			noise = np.random.normal(0,1)
			# print(noise)
			reward = q_estimates[move] + noise 
			avg_reward[move] += (reward - avg_reward[move]) / moves_count[move]
		# print(avg_reward)

		per_optimal_action.append(optimal_move_counter / step)
		rewards_till_now.append(reward) 

	# print(q_estimates)
	# print(rewards_till_now)
	return rewards_till_now, per_optimal_action

def nonstationary_ten_armed_testbed(eps, q_estimates , num_steps = total_steps,  initial_reward = 0, nonstationary = False):
	q_estimates = np.asarray(q_estimates)
	optimal_move = np.argmax(q_estimates) 
	moves_count = np.zeros(10) 
	rewards_till_now = [] 
	reward = 0 
	avg_reward = np.zeros(10) + initial_reward
	per_optimal_action = [] 
	optimal_move_counter = 0
	def get_best_move(array):
		return np.argmax(array) 
	
	
	alpha = 10
	for step in range(1,num_steps+1):
		if nonstationary:
			q_estimates += np.random.normal(0,0.01,10)
		epsilon = random.random() 
		if epsilon > eps: 
			best_move = get_best_move(avg_reward) 
			moves_count[best_move] += 1
			if best_move == np.argmax(q_estimates):
				optimal_move_counter += 1
			noise = sp.random.standard_normal() / 10
			reward = q_estimates[best_move] + noise 
			avg_reward[best_move] += (reward - avg_reward[best_move]) / alpha
		else:
			move = random.randint(0,9) 
			moves_count[move] += 1
			if move == np.argmax(q_estimates) :
				optimal_move_counter += 1
			noise = sp.random.standard_normal()
			reward = q_estimates[move] + noise 
			avg_reward[move] += (reward - avg_reward[move]) / alpha
		
		per_optimal_action.append(optimal_move_counter / step)
		rewards_till_now.append(reward) 

	return rewards_till_now, per_optimal_action


def UCBstationary_ten_armed_testbed(eps, q_estimates , num_steps = total_steps,c = 2, nonstationary = False):
	q_estimates = np.asarray(q_estimates) 
	optimal_move = np.argmax(q_estimates) 
	moves_count = np.zeros(10)
	rewards_till_now = [] 
	reward = 0 
	avg_reward = np.zeros(10)
	per_optimal_action = [] 
	optimal_move_counter = 0
	def get_best_move(array):
		return np.argmax(array) 
	
	for step in range(1,num_steps+1):
		epsilon = random.random() 

		if nonstationary:
			q_estimates += np.random.normal(0,0.01,10)
		if epsilon > eps: 
			best_move = get_best_move(np.array([x + c * sqrt(log(step) / (moves_count[i] + 1)) for i,x in enumerate(avg_reward)])) 
			moves_count[best_move] += 1
			if best_move == np.argmax(q_estimates) :
				optimal_move_counter += 1

			noise = sp.random.standard_normal()
			reward = q_estimates[best_move] + noise 
			avg_reward[best_move] += (reward - avg_reward[best_move]) / moves_count[best_move]
		else:
			move = random.randint(0,9) 
			if move == np.argmax(q_estimates) :
				optimal_move_counter += 1
			moves_count[move] += 1
			noise = sp.random.standard_normal()
			reward = q_estimates[move] + noise 
			avg_reward[move] += (reward - avg_reward[move]) / moves_count[move]

		per_optimal_action.append(optimal_move_counter / step)
	
		rewards_till_now.append(reward) 

	return rewards_till_now, per_optimal_action


def UCBnonstationary_ten_armed_testbed(eps, q_estimates , num_steps = total_steps,c = 2, nonstationary = False):
	optimal_move = np.argmax(q_estimates) 
	moves_count = np.zeros(10)
	rewards_till_now = [] 
	reward = 0 
	avg_reward = np.zeros(10)
	per_optimal_action = [] 
	optimal_move_counter = 0
	alpha = 10
	def get_best_move(array):
		return np.argmax(array) 
	
	for step in range(1,num_steps+1):
		if nonstationary:
			q_estimates += np.random.normal(0,0.01,10)

		epsilon = random.random() 
		if epsilon > eps: 
			best_move = get_best_move(np.array([x + c * sqrt(log(step) / (moves_count[i] + 1)) for i,x in enumerate(avg_reward)])) 
			moves_count[best_move] += 1
			if best_move == np.argmax(q_estimates):
				optimal_move_counter += 1

			noise = sp.random.standard_normal()
			reward = q_estimates[best_move] + noise 
			avg_reward[best_move] += (reward - avg_reward[best_move]) / alpha
		else:
			move = random.randint(0,9) 
			if move == np.argmax(q_estimates):
				optimal_move_counter += 1
			moves_count[move] += 1
			noise = sp.random.standard_normal()
			reward = q_estimates[move] + noise 
			avg_reward[move] += (reward - avg_reward[move]) / alpha

		per_optimal_action.append(optimal_move_counter / step)
	
		rewards_till_now.append(reward) 

	return rewards_till_now, per_optimal_action

if __name__ =='__main__':
	num_trials = 200
	# fig , (ax1,ax2) = plt.subplots(nrows = 2, ncols = 1)
	eps = [0.1]
	ckpts = [i for i in range(1,total_steps + 1) if i % step_size == 0] 
	# Q_estimates =  np.zeros(10)
	# for e in eps:
	# 	avg_trail = np.zeros(total_steps // step_size) 
	# 	avg_optimality = np.zeros(total_steps // step_size)
	# 	for _ in range(num_trials):
	# 		print('{}/{} trials done'.format(_,num_trials), end = '\r')
	# 		rewards, optimality = nonstationary_ten_armed_testbed(e, np.zeros(10), nonstationary = True)
	# 		avg_trail += np.asarray(rewards) 
	# 		avg_optimality += np.asarray(optimality)
	# 	avg_trail /= num_trials
	# 	avg_optimality /= num_trials 
	# 	ax1.plot(ckpts, avg_trail, label = 'epsilon = ' + str(e))
	# 	ax1.set_ylabel('Average Reward')
	# 	ax2.plot(ckpts, avg_optimality, label = 'epsilon = ' + str(e)) 
	# 	ax2.set_ylabel('% Optimal Action')
	# plt.title('Non Stationary Constant Step Size (Exercise 2.5)')
	# plt.legend() 
	# plt.show()  



	avg_optimistic = np.zeros(total_steps // step_size) 
	avg_notoptimistic = np.zeros(total_steps // step_size)  
	avg_ucb = np.zeros(total_steps // step_size) 
	for _ in range(num_trials):
		__ , optmistic =  stationary_ten_armed_testbed(0, np.zeros(10)  + np.random.normal(0,0.01,10) , initial_reward = 5, nonstationary = False)
		__ , not_optmistic = stationary_ten_armed_testbed(0.1, np.zeros(10)+ np.random.normal(0,0.01,10) , nonstationary = False) 
		__ , ucb = UCBstationary_ten_armed_testbed(0,np.zeros(10)+ np.random.normal(0,0.01,10) , nonstationary = False)
		avg_optimistic += np.asarray(optmistic) 
		avg_notoptimistic += np.asarray(not_optmistic) 
		avg_ucb += np.asarray(ucb)

	avg_notoptimistic /= num_trials 
	avg_optimistic /= num_trials
	avg_ucb /= num_trials

	plt.plot(ckpts,avg_optimistic, label = 'Q_1 = 5, e = 0')
	plt.plot(ckpts,avg_notoptimistic, label = 'Q_1 = 0, e = 0.1')
	plt.plot(ckpts,avg_ucb,label = 'UCB, c = 2')
	plt.title('Optmistic Greedy Vs Realistic E Greedy Vs UCB (Fig 2.3)')
	plt.ylabel('% Optimal Action')
	plt.legend()
	plt.show()


	


