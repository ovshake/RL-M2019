import numpy as np 
import matplotlib.pyplot as plt 







def random_walk(v_pie,alpha):
	cur_state = 3
	walk = [cur_state]
	while cur_state != 6 and cur_state != 0:
		action = np.random.choice([1,-1]) 
		next_state = cur_state + action
		reward = 0
		if next_state == 6:
			reward = 1 

		v_pie[cur_state] += alpha * (reward + v_pie[next_state] - v_pie[cur_state]) 
		cur_state = next_state 

	return v_pie 



def TD_0_monte_Carlo_6_1_left(num_episodes, breaks):
	v_pie = np.ones(7) / 2
	v_pie[6] = v_pie[0] = 0
	alpha = 0.1 
	plt.plot([i for i in range(7)] ,  v_pie, label = 'Before Iterations') 
	for i in range(1,num_episodes+1):
		print(i)

		v_pie = random_walk(v_pie, alpha) 
		if i in breaks:
			plt.plot([i for i in range(7)] ,  np.asarray(v_pie) , label = 'After {} Iterations'.format(i)) 

	plt.plot([i for i in range(7)] , [i/6 for i in range(7)] , label = 'True Values') 
	plt.legend() 
	plt.xlabel('State')
	plt.ylabel('Value')
	plt.title('Figure 6.2')
	plt.savefig('Figure 6.2.jpeg') 

def monte_carlo(num_episodes , alpha):
	true_vals = [0] + [i/6 for i in range(1,6)] + [0]
	rmse_array = np.zeros(num_episodes) 
	v_pie = np.ones(7) 
	v_pie[0] = v_pie[6] = 0 
	for i in range(num_episodes):
		cur_state = 3
		player_trajecory = [] 
		while cur_state != 6 and cur_state != 0:
			action = np.random.choice([1,-1])
			next_state = cur_state + action 
			reward = 0
			if next_state == 6:
				reward = 1
			player_trajecory.append((cur_state,action, reward)) 
			cur_state = next_state 
		G = 0
		for c,a,r in player_trajecory[::-1]:
			G += r 
			v_pie[c] += alpha * (G - v_pie[c])
			rmse = np.sqrt(np.power(true_vals -  v_pie , 2).mean()) 
			rmse_array[i] = rmse 

	return rmse_array  

		
def td_0(num_episodes,alpha):
	true_vals = [0] + [i/6 for i in range(1,6)] + [0]
	v_pie = np.ones(7) / 2
	v_pie[6] = v_pie[0] = 0
	alpha = 0.1 
	error_episodes = np.zeros(num_episodes)
	for i in range(num_episodes):
		v_pie = random_walk(v_pie, alpha) 
		rmse = np.sqrt(np.power(true_vals - v_pie , 2).mean())
		error_episodes[i] = rmse 

	return error_episodes 

def runs_montecarlo(num_runs, alpha):
	num_episodes = 100
	run_error = np.zeros((num_runs,num_episodes)) 

	for i in range(num_runs):
		rmse_array = monte_carlo(num_episodes, alpha) 
		run_error[i] = rmse_array

	mean = np.mean(run_error , axis = 0)
	return mean 

def runs_td(num_runs, alpha):
	num_episodes = 100 
	run_error = np.zeros((num_runs, num_episodes)) 
	for i in range(num_runs):
		error_episodes = td_0(num_episodes, alpha) 
		run_error[i] = error_episodes 

	mean = np.mean(run_error, axis = 0) 
	return mean 

 

if __name__ == '__main__':
	alpha_monte = [0.1, 0.2, 0.3,  0.4] 
	alpha_td0 = [.15, .05, .1]
	for alpha_ in alpha_monte + alpha_td0:
		print(alpha_)
		if alpha_ in alpha_monte:
			mean_array = runs_montecarlo(100,alpha_)
			plt.plot([i for i in range(100)], mean_array, label = 'Monte Carlo a = '+str(alpha_)) 
		if alpha_ in alpha_td0:
			mean_array = runs_td(100,alpha_)
			plt.plot([i for i in range(100)], mean_array, label = 'TD(0) a = '+str(alpha_)) 

	plt.xlabel('Walks/Episode')
	plt.ylabel('RMS Error')
	plt.legend() 
	plt.savefig('Example 6.2 (Right).jpeg')

















if __name__ == '__main__':
	TD_0_monte_Carlo_6_1_left(100, [0,1,10,100]) 







