# Monte-Carlo-Simulation-RL 
     Agents generates samples and Learns​
     There is no model (model free)​
     Value = avg return​
     Only after completing a episode values are updated ​
     Π(s)-->avg. Q(s,a)
## Tyepes of Monte Carlo 
# 1. First Visit Monte Carlo 

![Screenshot from 2025-04-03 12-43-55](https://github.com/user-attachments/assets/4f8acfa0-84d3-4270-9503-e130b2fc2190)


# 2. Every Visit Monte Carlo 

![Screenshot from 2025-04-03 12-44-05](https://github.com/user-attachments/assets/7c91d0c9-c29e-462a-8c46-c51fa0513d0c)


 ![Screenshot from 2025-04-03 12-31-33](https://github.com/user-attachments/assets/c4b1413c-8221-4f64-b605-586e2ad7baed)


# Here a baic 3*3 grid example has been solved using monte carlo 
Environment Understanding
  The robot starts in the top-left corner (position 0,0)
  Most cells have a -1 penalty
  Bottom-middle cell has a -10 penalty
  Bottom-right cell has the +10 reward goal
  The task is to find the optimal policy to reach the goal with maximum cumulative reward

# Monte Carlo Methods Implementation
I've implemented both First-Visit and Every-Visit Monte Carlo methods for this problem:

# First-Visit Monte Carlo: Only considers the first occurrence of each state in an episode when updating values
# Every-Visit Monte Carlo: Updates values based on every occurrence of a state in an episode

Both methods:
 Use exploration with epsilon-greedy policy
 Learn by generating complete episodes (robot's path from start to goal)
 Update state values based on actual returns observed during episodes
 Gradually improve the policy by making it greedy with respect to the estimated values

# Results Analysis
Both methods converge to similar solutions, showing that the optimal path is: 
  
  Start at (0,0)
  Move right to (0,1)
  Move right to (0,2)
  Move down to (1,2)
  Move down to (2,2) to reach the goal

![Screenshot from 2025-04-03 11-52-47](https://github.com/user-attachments/assets/61c86ae7-76c6-4b44-84e5-62ea4ffd8f46)


![Screenshot from 2025-04-03 12-22-50](https://github.com/user-attachments/assets/8b3cf8af-786e-4eb4-bf77-5fdc631b2bd5)
![Figure_1](https://github.com/user-attachments/assets/be2f286f-0a5a-495e-8e09-5c8db948255d)
