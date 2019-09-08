import numpy as np 

def lyingOutside(x,y):
    return x < 0 or x >= 5 or y < 0 or y >= 5


if __name__ == '__main__':
    v_pie = np.zeros((5,5)) 
    num_iters = 1000 
    gamma = 0.9
    for it in range(num_iters):
        print("Iterations: {}/{}".format(it,num_iters), end = '\r')
        for x in range(5):
            for y in range(5):
                if x == 0 and y == 1:
                    v_pie[x][y] = 10 + gamma * v_pie[4][1]
                    continue 
                elif x == 0 and y == 3:
                    v_pie[x][y] = 5 + gamma * v_pie[2][3]
                    continue 
                
                nX = [-1,0,1,0]
                nY = [0,1,0,-1]
                max_reward = []  
                for i in range(4):
                    u = x + nX[i] 
                    v = y + nY[i] 
                    if lyingOutside(u,v):
                        reward = -1 + gamma * v_pie[x][y] 
                        max_reward.append(reward) 
                    else:
                        reward = gamma * v_pie[u][v] 
                        max_reward.append(reward) 
                v_pie[x][y] = max(max_reward)
    print(v_pie) 

                 
                    
