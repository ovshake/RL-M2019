import numpy as np 

def lyingOutside(x,y):
    return x < 0 or x >= 4 or y < 0 or y >= 4

if __name__ == '__main__':
    LEFT = 0 
    UP = 1
    DOWN = 2
    RIGHT = 3
    v_opt = np.zeros((4,4))
    pie_opt = np.zeros((4,4),dtype=int)  
    num_iters = 1000 
    convergence_criteria = 5
    stable = True
    theta = -np.inf
    itr = 0
    p = 1/4
    nX = [0,-1,1,0]
    nY = [-1,0,0,1]
    while 1:
        #Policy Evaluation
        print('Iterations done: {}'.format(itr),end = '\n')
        itr += 1
        while 1:
            for x in range(4):
                for y in range(4):
                    if (x == 0 and y == 0) or (x == 3 and y == 3):
                        continue 
                    temp = v_opt[x][y]
                    opt_x = x+nX[pie_opt[x][y]]
                    opt_y = y+nY[pie_opt[x][y]]
                    if lyingOutside(opt_x,opt_y):
                        v_opt[x][y] = (-1 + v_opt[x][y])
                    else:
                        v_opt[x][y] = (-1 + v_opt[opt_x][opt_y])
                    theta = max(theta , abs(temp - v_opt[x][y])) 
    
                    print(theta)
            if theta < convergence_criteria:
                break
        print('-------------------------')
        stable = True
        #Policy Improvement
        for x in range(4):
            for y in range(4):
                if (x == 0 and y == 0) or (x == 3 and y == 3):
                    continue 
                rewards = [] 
                actions = [] 
                for i in range(4):
                    u = x + nX[i] 
                    v = y + nY[i] 
                    if lyingOutside(u,v):
                        rewards.append(-1 + v_opt[x][y])
                        actions.append(i)
                    else:
                        rewards.append(-1 + v_opt[u][v]) 
                        actions.append(i) 
                    print('location {} {}: Neighbour: {} {} : Rewards: {}'.format(x,y,u,v,rewards)) #Printing the iteration to show improvement
                max_reward = np.max(rewards) 
                best_action = np.argmax(rewards) 
                if best_action != pie_opt[x][y]:
                    stable = False 
                pie_opt[x][y] = best_action 
        
        print(v_opt)
        print(pie_opt)
        if stable:
            break 

print("Final Policy {} Final V* {} ".format(pie_opt, v_opt))

#The tie is broken by np.argmax; this is the bug fix