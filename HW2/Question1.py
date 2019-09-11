import numpy as np 

def lyingOutside(x,y):
    return x < 0 or x >= 5 or y < 0 or y >= 5

if __name__ == '__main__':
    P = np.zeros((25,25))
    G = np.zeros(25) 
    I = np.eye(25)
    gamma = 0.9
    for x in range(5):
        for y in range(5):
            if x == 0 and y == 1:
                P[x*5 + y][4*5 + 1] = 1
                G[x*5 + y] = 10 
            elif x == 0 and y == 3:
                P[3][13] = 1
                G[3] = 5
            else:
                nX = [-1,0,1,0]
                nY = [0,1,0,-1]
                for i in range(4):
                    if lyingOutside(x+nX[i],y+nY[i]):
                        P[x*5 + y][x*5 + y] += 1/4 
                        G[x*5 + y]  += (0.25 * -1) #The expected value of the policy 
                    else:
                        P[x*5 + y][(x+nX[i])*5 + (y+nY[i])] = 1/4

    # print(G) 
    v_pie = np.matmul(np.linalg.inv(I - gamma * P),G)
    print(v_pie.reshape(5,5)) #final policy



    


