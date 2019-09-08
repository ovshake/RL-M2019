import numpy as np 

def lyingOutside(x,y):
    return x < 0 or x >= 4 or y < 4 or y >= 4

if __name__ == '__main__':
    LEFT = 0 
    UP = 1
    DOWN = 2
    RIGHT = 3
    v_opt = np.zeros((4,4))
    pie_opt = np.zeros((4,4),dtype=int)  
    num_iters = 1000 
    convergence_criteria = 1e-4 
    stable = True
    theta = 1e10 
    itr = 0
    nX = [0,-1,1,0]
    nY = [-1,0,0,1]
    while 1:
        print('Iterations done: {}'.format(itr),end = '\r')
        itr += 1
        # while theta > convergence_criteria:
        #     print(theta)
        for _ in range(1000):
            theta = -1
            for x in range(4):
                for y in range(4):
                    if (x == 0 and y == 0) or (x == 3 and y == 3):
                        continue 
                    temp = v_opt[x][y]
                    opt_x = x+nX[pie_opt[x][y]]
                    opt_y = y+nY[pie_opt[x][y]]
                    if lyingOutside(opt_x,opt_y):
                        v_opt[x][y] = -1 + v_opt[x][y]
                    else:
                        v_opt[x][y] = -1 + v_opt[opt_x][opt_y]
                    theta = max(theta, abs(v_opt[x][y] - temp)) 
                    # print(x,y)
        print('-------------------------')
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
                        rewards(-1 + v_opt[u][v]) 
                        actions.append(i) 
                
                max_reward = np.max(rewards) 
                best_action = np.argmax(rewards) 
                if best_action != pie_opt[x][y]:
                    stable = False 
                pie_opt[x][y] = best_action 
        
        if stable:
            break 
