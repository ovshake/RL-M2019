import numpy as np
from numpy.random import poisson as p 
from scipy.stats import poisson
from sys import exit 

possion_dict = {}
def get_p_cdf(x,mu):
    key = str(x) + ' ' +str(mu) 
    if key not in possion_dict:
        possion_dict[key] = poisson.pmf(x,mu) 
    return possion_dict[key]

def isValid(x,y,i):
    return (0 <= i <= x) or (-y <= i <= 0)

max_cars = 20
first_rental = 3 
second_rental = 4 
first_return = 3
second_return = 4
gamma = 0.9
movement_cost = 2
parking_cost = 4
theta = -np.inf
v_opt = np.zeros((max_cars + 1,max_cars + 1)) 
pie_opt = np.zeros((max_cars + 1,max_cars + 1), dtype = int)  
upper_bound = 11
actions = np.arange(-max_cars,max_cars + 1,dtype=int)
rental_money = 10
conv_criteria = .1 


def calculate_return(x,y, policy):
    global v_opt, pie_opt
    ret = 0
    # if policy > 1:
    #     ret -= movement_cost * abs(policy - 1)
    # else:
    ret -= movement_cost * abs(policy)
    
    for f_rent in range(max_cars + 1):
        for f_ret in range(max_cars + 1):
            for s_rent  in range(max_cars + 1):
                for s_ret in range(max_cars + 1):
                    # print('Current Poss: {} {} {} {}'.format(f_rent, f_ret, s_rent, s_ret))
                    p_f_rent = get_p_cdf(f_rent, first_rental) 
                    p_f_ret = get_p_cdf(f_ret, first_return) 
                    p_s_rent = get_p_cdf(s_rent, second_rental) 
                    p_s_ret = get_p_cdf(s_ret, second_return) 
                    total_p = p_f_rent * p_f_ret * p_s_rent * p_s_ret
                    CARS_AT_FIRST_LOC = max(x - policy, max_cars) 
                    CARS_AT_SEC_LOC = max(y + policy, max_cars) 
                    # print(CARS_AT_FIRST_LOC, CARS_AT_SEC_LOC, "BAC")
                    possible_rentals_first_loc = min(CARS_AT_FIRST_LOC, f_rent)
                    possible_rentals_second_loc = min(CARS_AT_SEC_LOC, s_rent)
                    # print(CARS_AT_FIRST_LOC, CARS_AT_SEC_LOC, "1")
                    reward = (possible_rentals_first_loc + possible_rentals_second_loc) * rental_money
                    CARS_AT_FIRST_LOC -= possible_rentals_first_loc 
                    CARS_AT_SEC_LOC -= possible_rentals_second_loc 
                    CARS_AT_FIRST_LOC = min(CARS_AT_FIRST_LOC + f_ret, max_cars)
                    CARS_AT_SEC_LOC = min(CARS_AT_SEC_LOC + s_ret, max_cars) 
                    print(CARS_AT_FIRST_LOC, CARS_AT_SEC_LOC, "2")
                    if CARS_AT_FIRST_LOC > max_cars // 2:
                        reward -= 10 
                    if CARS_AT_SEC_LOC > max_cars // 2:
                        reward -= 10
                    
                    
                    ret += total_p * (reward + gamma * v_opt[CARS_AT_FIRST_LOC,CARS_AT_SEC_LOC])
                    

    
    return ret 


if __name__ == '__main__':
    state = np.zeros((max_cars+1,max_cars+1))
    while 1:
        while 1:
            print('Starting Iterations')
            theta = -np.inf
            for x in range(max_cars):
                for y in range(max_cars):
                    print("Current Coordinate: {} {}".format(x,y))
                    temp = v_opt[x][y] 
                    ret = calculate_return(x,y,pie_opt[x][y]) 
                    v_opt[x][y] = ret
                    theta = max(theta , abs(temp - v_opt[x][y])) 
            
            print(theta)
            if theta < conv_criteria:
                break 
        
        policy_stable = True 
        for x in range(max_cars + 1):
            for y in range(max_cars + 1):
                print("Current Policy Improv State: {} {}".format(x,y))
                returns_from_all_actions = [] 
                for i in actions:
                    # print("Action: {}".format(i))
                    if isValid(x,y,i):
                        returns_from_all_actions.append(calculate_return(x,y,i)) 
                    else:
                        returns_from_all_actions.append(-np.inf) 
                temp = pie_opt[x][y] 
                pie_opt[x][y] = actions[np.argmax(returns_from_all_actions)] 
                if pie_opt[x][y] != temp:
                    policy_stable = False 
        
        if policy_stable:
            break 
    
    print(pie_opt)






                




                                    

                                     


                                     



    
