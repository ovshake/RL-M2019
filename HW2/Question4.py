import numpy as np 
from scipy.optimize import linprog 

def lyingOutside(x,y):
    return x < 0 or x >= 5 or y < 0 or y >= 5

def encode_dir(x,y,d):
    return 5*4*x + y*4 + d

def encode(x,y):
    return 5*x + y

nX = [0,-1,1,0]
nY = [-1,0,0,1]
if __name__ == '__main__':
    v_pie = np.zeros((5,5))  # this will store my final optimal value function 
    Q = np.zeros(100) # This is the rewards for each state
    gamma = 0.9
    left = 0 
    up = 1
    right = 3
    down = 2
    P = np.zeros((100,25)) 
    for x in range(5): #for each state and for all the directions 
        for y in range(5):
            for i in range(4):
                if x == 0 and y == 1:
                    P[encode_dir(x,y,i)][encode(x,y)] -= 1
                    P[encode_dir(x,y,i)][encode(4,1)] += gamma 
                    Q[encode_dir(x,y,i)] -= 10
                elif x == 0 and y == 3:
                    P[encode_dir(x,y,i)][encode(x,y)] -= 1 
                    P[encode_dir(x,y,i)][encode(2,3)] += gamma
                    Q[encode_dir(x,y,i)] -= 5 
    

                elif lyingOutside(x+nX[i],y+nY[i]):
                    P[encode_dir(x,y,i)][encode(x,y)] += gamma - 1
                    Q[encode_dir(x,y,i)] -= 1 
                
                else:
                    P[encode_dir(x,y,i)][encode(x+nX[i],y+nY[i])] += gamma 
                    P[encode_dir(x,y,i)][encode(x,y)] -= 1
                    
    

    x = linprog(np.ones(25),P,Q) #Solving the inequality\
    v_pie = (x.x).reshape(5,5)
    print(v_pie) #final optimal V* 
    pie_star = np.zeros((5,5))
    for x in range(5):
        for y in range(5):
            v_neighbours = []
            for i in range(4):
                if (lyingOutside(x+nX[i],y+nY[i])):
                   v_neighbours.append(v_pie[x][y])
                else:
                    v_neighbours.append(v_pie[x+nX[i]][y+nY[i]]) 
            pie_star[x,y] = np.argmax(v_neighbours) 

    print(pie_star) #this is the optimal policy  



    
    



    

                 
                    
