# Policy-Iteration

The state-space is a 10×10 grid, cells that are obstacles are marked in gray. The initial state of the robot is in blue and our desired terminal state is in green. 

<img width="318" alt="Screenshot 2023-08-08 at 7 48 03 AM" src="https://github.com/RenuReddyK/Policy-Iteration/assets/68454938/e3fd4f77-7b42-4131-b0b4-09c0fe35fafb">

The robot gets a reward of 10 if it reaches the desired terminal state with a discount factor of 0.9. At each non-obstacle cell, the robot can attempt to move to any of the immediate neighboring cells using one of the four controls (North, East, West and South). The robot cannot diagonally. 

The move succeeds with probability 0.7 and with remainder probability 0.3 the robot can end up at some other cell as follows:
P(moves north | control is north) = 0.7, P(moves west | control is north) = 0.1, P(moves east | control is north) = 0.1, P(does not move | control is north) = 0.1.

Similarly, if the robot desired to go east, it may end up in the cells to its north, south, or stay put at the original cell with total probability 0.3 and actually move to the cell east with probability 0.7. The cost pays a cost of 1 (i.e., reward is -1) for each control input it takes, regardless of the outcome. If the robot ends up at a state marked as an obstacle, it gets a reward of -10 for each time-step that it remains inside the obstacle cell. The robot is allowed to stay in the goal state indefinitely (i.e., take a special action to “not move”) and this action gets no reward/cost.

Implemented policy iteration to find the best trajectory for the robot to go from the blue cell to the green cell.
