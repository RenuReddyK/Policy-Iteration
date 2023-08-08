import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import colors
#Policy Iteration
#Considering the following Markov Decision Process:
#State-space --> 10×10 grid
#The robot gets a reward of 10 if it reaches the desired terminal state with a discount factor of 0.9.
#At each non-obstacle cell, the robot can attempt to move to any of the immediate neighboring cells using one of the four controls (North, East, West and South). The robot cannot diagonally. 
#The move succeeds with probability 0.7 and with remainder probability 0.3 the robot can end up at some other cell as follows:
#P(moves north | control is north) = 0.7,
#P(moves west | control is north) = 0.1,
#P(moves east | control is north) = 0.1,
#P(moves south | control is north) = 0,
#P(does not move | control is north) = 0.1.
#At boundaries assuming there is equal probability to move along the feasible states 
# --> 3 possibilities with 0.33 at corners and 2 possibilities with 0.5 at edge states.

#Policy iteration to find the best trajectory for the robot to go from the start to the goal.

#Transition matrix --> (10,10,10,10)
def Down(T, N, goal, obstacles):
    for i in range(N):
        for j in range(N):
            # When transitioning to goal state
            if [i,j] == goal:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1 
            # Sticky obstacles
            elif [i,j] in obstacles:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1 
            else:
                # General case:
                if i+1 <= N-1 and j+1 <= N-1 and i-1 >= 0 and j-1>=0 :
                    T[i, j, i+1, j] = 0.7
                    T[i, j, i, j] = 0.1
                    T[i, j, i, j+1] = 0.1
                    T[i, j, i, j-1] = 0.1
    return T

def Up(T, N, goal, obstacles):
    for i in range(N):
        for j in range(N):
            # When transitioning to goal state
            if [i,j] == goal:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1
            # Sticky obstacles
            elif [i,j] in obstacles:
                T[i, j, i, j] = 1 
            else:
                # General case:
                if i+1 <= N-1 and j+1 <= N-1 and i-1>= 0 and j-1 >= 0:
                    T[i, j, i-1, j] = 0.7
                    T[i, j, i, j] = 0.1
                    T[i, j, i, j+1] = 0.1
                    T[i, j, i, j-1] = 0.1
                
    return T

def Right(T,N, goal, obstacles):
    for i in range(N):
        for j in range(N):
            # When transitioning to goal state
            if [i,j] == goal:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1
            # Sticky obstacles
            elif [i,j] in obstacles:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1 
            else:
                # General case:
                if i+1 <= N-1 and j+1 <= N-1 and i-1 >= 0:
                    T[i, j, i-1, j] = 0.1
                    T[i, j, i, j] = 0.1
                    T[i, j, i, j+1] = 0.7
                    T[i, j, i+1, j] = 0.1
                    
    return T
        
def Left(T,N, goal, obstacles):
    for i in range(N):
        for j in range(N):
            # When transitioning to goal state
            if [i,j] == goal:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1
            # Sticky obstacles
            elif [i,j] in obstacles:
                T[i, j, :, :] = 0
                T[i, j, i, j] = 1 
            else:
                # General case:
                if i+1 <= N-1 and j-1 >= 0 and i-1 >= 0:
                    T[i, j, i-1, j] = 0.1
                    T[i, j, i, j] = 0.1
                    T[i, j, i, j-1] = 0.7
                    T[i, j, i+1, j] = 0.1
    return T

def transition_mat(N, action):
    obstacles = Obstacle_positions(N)
    goal = [8,8]
    T = np.zeros((N,N,N,N))
    if action == 0: #(0,1):          
        Right(T,N,goal, obstacles)
    if action == 1: #(0,-1):           
        Left(T,N,goal, obstacles)  
    if action == 2: #(1, 0):            
        Up(T,N,goal, obstacles)
    if action == 3: #(-1,0):             
        Down(T,N,goal, obstacles)
    return T

def Terminal_cost(goal):
    N =10
    Terminal_cost = np.zeros((N,N,1))
    [i,j] = goal
    Terminal_cost[i,j] = 10
    return Terminal_cost

def Obstacle_positions(N):
    obstacles = [] 
    for i in range(N):
        for j in [0,N-1]:
            obstacles.append([i,j])
    for i in [0,N-1]:
        for j in range(N):
            obstacles.append([i,j])
    obstacles.append([7,3])
    obstacles.append([7,4])
    obstacles.append([7,5])
    obstacles.append([7,6])
    obstacles.append([5,4])
    obstacles.append([4,4])
    obstacles.append([3,4])
    obstacles.append([2,4])
    obstacles.append([2,5])
    obstacles.append([4,7])
    obstacles.append([5,7])
    obs_set = []
    [obs_set.append(x) for x in obstacles if x not in obs_set]
    # print("obs",obs_set)
    return obs_set

def Runtime_cost(N):
    obstacles = Obstacle_positions(10)
    Runtime_cost = -1*np.ones((N,N,4))
    for obstacle in obstacles:
        [i,j] = obstacle
        Runtime_cost[i,j,:] = -10
    Runtime_cost[8,8,:] = 10
    # For Plotting intial enviroment:
    # Runtime_cost[8,8,u] = -20
    # Runtime_cost[3,3,u] = -30
    return Runtime_cost

def policy_evaluation(Discount_factor, action, N):
    #Policy Evaluation:
    for i in range(10):
        for j in range(10):
            Q = Runtime_cost(N)
            T = transition_mat(N, int(action[i,j]))
            I = np.identity(N*N)
            T = T.reshape((100, 100))
            J_initial = np.linalg.inv(I -  Discount_factor * T) @ Q[:,:,int(action[i,j])].reshape(N*N,1)
            J_initial = J_initial.reshape((10,10))
    return J_initial

#Plotting intial enviroment
# colormap = colors.ListedColormap(["blue", "green",  "black", "white"])
# fig = pyplot.figure(figsize=(5,5))
# ax = fig.gca()
# Q = Runtime_cost(10,0)[:,:,0]
# ax.set_xticks(np.arange(0.5, 10.5, 1))
# ax.set_yticks(np.arange(0.5, 10.5, 1))
# pyplot.imshow(Q, alpha=0.75, cmap=colormap)
# pyplot.grid()
# pyplot.show()

# Plotting Value function
fig = pyplot.figure(figsize=(5,5))
ax = fig.gca()
V = policy_evaluation(Discount_factor=0.9, action =np.zeros((10,10)), N=10)
ax.set_xticks(np.arange(0.5, 10.5, 1))
ax.set_yticks(np.arange(0.5, 10.5, 1))
pyplot.imshow(V)
pyplot.grid()
pyplot.show()

#Policy Improvement:
def Policy_improvement(Discount_factor, N, J_i):
    u = np.ones((10, 10))
    for i in range(N):
        for j in range(N):
            act=0
            T = transition_mat(N, act)
            Q = Runtime_cost(N)
            R =  Q[i,j,act] + Discount_factor * T[i,j,:,:].reshape((1,100)) @ J_i.reshape((100,1))

            act=1
            T = transition_mat(N, act)
            Q = Runtime_cost(N)
            L = Q[i,j,act] + Discount_factor * T[i,j,:,:].reshape((1,100)) @ J_i.reshape((100,1))

            act=2
            T = transition_mat(N, act)
            Q = Runtime_cost(N)
            U = Q[i,j,act] + Discount_factor * T[i,j,:,:].reshape((1,100)) @ J_i.reshape((100,1))

            act=3
            T = transition_mat(N, act)
            Q = Runtime_cost(N)
            D = Q[i,j,act] + Discount_factor * T[i,j,:,:].reshape((1,100)) @ J_i.reshape((100,1))
            u[i, j] = np.argmax((R[0,0], L[0,0], U[0,0], D[0,0]))
    return u

def Policy_Iteration_full(Discount_factor, action, N, iterations):
    for iter in range(iterations+1):
        J = policy_evaluation(Discount_factor, action, N)
        u = Policy_improvement(Discount_factor, N, J_i = J)
        action = u
    return J, action

# Plotting feedback control
J_i, u = Policy_Iteration_full(Discount_factor = 0.9, action=np.zeros((10,10)), N =10, iterations=4)
fig = pyplot.figure(figsize=(5,5))
ax = fig.gca()
ax.set_xticks(np.arange(0.5, 10.5, 1))
ax.set_yticks(np.arange(0.5, 10.5, 1))
pyplot.imshow(J_i)
pyplot.grid()
for i in range(10):
    for j in range(10):
        obstacles = Obstacle_positions(10)
        goal = [8,8]
        if [i,j] not in obstacles and [i,j] != [8,8]:
            if u[i,j] == 0:
                pyplot.arrow(j, i, 0.25, 0, head_width=0.15)
            elif u[i,j] == 1:
                pyplot.arrow(j, i, -0.25, 0, head_width=0.15)
            elif u[i,j] == 2:
                pyplot.arrow(j, i, 0, -0.25, head_width=0.15)
            elif u[i,j] == 3:
                pyplot.arrow(j, i, 0, 0.25, head_width=0.15)
pyplot.show()
print("done")