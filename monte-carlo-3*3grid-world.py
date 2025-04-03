import numpy as np
import matplotlib.pyplot as plt
import random

# Define the environment
class GridWorld:
    def __init__(self):
        # Grid layout: (row, column)
        # 0: empty cell with -1 reward
        # 1: robot starting position (0 reward)
        # 2: high negative reward (-10)
        # 3: goal with positive reward (+10)
        self.grid = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 2, 3]
        ])
        
        self.rewards = {
            0: -1,    # Regular cell
            1: 0,     # Starting position
            2: -10,   # High penalty
            3: 10     # Goal
        }
        
        self.actions = ["up", "right", "down", "left"]
        self.start_pos = (0, 0)
        self.goal_pos = (2, 2)
        self.high_penalty_pos = (2, 1)
        
        # Terminal states
        self.terminal_states = [self.goal_pos]
        
        # Current position
        self.reset()
    
    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        row, col = self.current_pos
        
        # Move according to action
        if action == "up" and row > 0:
            row -= 1
        elif action == "right" and col < 2:
            col += 1
        elif action == "down" and row < 2:
            row += 1
        elif action == "left" and col > 0:
            col -= 1
            
        # Update position
        self.current_pos = (row, col)
        
        # Get cell type and reward
        cell_type = self.grid[row, col]
        reward = self.rewards[cell_type]
        
        # Check if terminal state
        done = self.current_pos in self.terminal_states
        
        return self.current_pos, reward, done

    def get_all_states(self):
        states = []
        for i in range(3):
            for j in range(3):
                states.append((i, j))
        return states

# Generate an episode using the given policy
def generate_episode(env, policy, max_steps=100):
    episode = []
    state = env.reset()
    
    for _ in range(max_steps):
        # Select action based on policy
        action = random.choice(policy[state])
        
        # Take action and observe next state and reward
        next_state, reward, done = env.step(action)
        
        # Store state, action, reward
        episode.append((state, action, reward))
        state = next_state
        
        if done:
            break
            
    return episode

# First-Visit Monte Carlo prediction
def first_visit_mc(env, policy, num_episodes=10000, gamma=0.9):
    # Initialize value function
    V = {}
    returns = {}
    
    for state in env.get_all_states():
        V[state] = 0.0
        returns[state] = []
    
    # Iterate through episodes
    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        
        # Keep track of visited states
        visited_states = set()
        
        # Calculate returns
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, _, reward = episode[t]
            G = gamma * G + reward
            
            # First-visit MC: only consider first occurrence of each state
            if state not in visited_states:
                visited_states.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    
    return V

# Every-Visit Monte Carlo prediction
def every_visit_mc(env, policy, num_episodes=10000, gamma=0.9):
    # Initialize value function
    V = {}
    returns = {}
    
    for state in env.get_all_states():
        V[state] = 0.0
        returns[state] = []
    
    # Iterate through episodes
    for _ in range(num_episodes):
        episode = generate_episode(env, policy)
        
        # Calculate returns
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state, _, reward = episode[t]
            G = gamma * G + reward
            
            # Every-visit MC: consider every occurrence of each state
            returns[state].append(G)
            V[state] = np.mean(returns[state])
    
    return V

# Monte Carlo Control (finding optimal policy)
def monte_carlo_control(env, method="first_visit", num_episodes=10000, gamma=0.9, epsilon=0.1):
    # Initialize action-value function and policy
    Q = {}
    returns = {}
    policy = {}
    
    for state in env.get_all_states():
        policy[state] = env.actions.copy()
        for action in env.actions:
            Q[(state, action)] = 0.0
            returns[(state, action)] = []
    
    # Iterate through episodes
    for episode_num in range(num_episodes):
        # Generate episode using epsilon-greedy policy
        episode = []
        state = env.reset()
        
        for _ in range(100):  # Max steps per episode
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                # Choose best action based on Q values
                q_values = [Q[(state, a)] for a in env.actions]
                max_q = max(q_values)
                best_actions = [a for a, q in zip(env.actions, q_values) if q == max_q]
                action = random.choice(best_actions)
            
            # Take action and observe next state and reward
            next_state, reward, done = env.step(action)
            
            # Store state, action, reward
            episode.append((state, action, reward))
            state = next_state
            
            if done:
                break
        
        # Process episode and update Q values
        visited_state_actions = set()
        G = 0
        
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            state_action = (state, action)
            G = gamma * G + reward
            
            if method == "first_visit":
                # First-visit MC: only consider first occurrence of each state-action pair
                if state_action not in visited_state_actions:
                    visited_state_actions.add(state_action)
                    returns[state_action].append(G)
                    Q[state_action] = np.mean(returns[state_action])
            else:  # every_visit
                # Every-visit MC: consider every occurrence of each state-action pair
                returns[state_action].append(G)
                Q[state_action] = np.mean(returns[state_action])
        
        # Update policy (greedy with respect to Q)
        for state in env.get_all_states():
            q_values = [Q[(state, a)] for a in env.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(env.actions, q_values) if q == max_q]
            
            # Set policy to be greedy with respect to Q
            policy[state] = best_actions
    
    return Q, policy

# Visualize the results
def visualize_policy(env, policy):
    # Define arrow symbols for actions
    arrows = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
    
    # Create a grid for visualization
    grid = np.zeros((3, 3), dtype=object)
    
    # Fill in grid with policy directions
    for i in range(3):
        for j in range(3):
            state = (i, j)
            
            # For terminal states, use special symbols
            if state == env.goal_pos:
                grid[i, j] = "+10"
            elif state == env.high_penalty_pos:
                grid[i, j] = "-10"
            else:
                # Get the best action(s) for this state
                best_actions = policy[state]
                if len(best_actions) == 1:
                    grid[i, j] = arrows[best_actions[0]]
                else:
                    # If multiple best actions, show first one followed by +
                    grid[i, j] = arrows[best_actions[0]] + "+"
    
    # Set robot at start position
    if grid[env.start_pos] == 0:
        grid[env.start_pos] = "R"
    else:
        grid[env.start_pos] = "R," + grid[env.start_pos]
    
    return grid

# Visualize the value function
def visualize_values(env, V):
    # Create a grid for visualization
    grid = np.zeros((3, 3))
    
    # Fill in grid with state values
    for i in range(3):
        for j in range(3):
            state = (i, j)
            grid[i, j] = V[state]
    
    return grid

# Run the algorithm and visualize results
def main():
    env = GridWorld()
    
    # Initial random policy
    initial_policy = {}
    for state in env.get_all_states():
        initial_policy[state] = env.actions
    
    print("Running First-Visit Monte Carlo Control...")
    Q_first, policy_first = monte_carlo_control(env, method="first_visit", num_episodes=50000)
    
    print("Running Every-Visit Monte Carlo Control...")
    Q_every, policy_every = monte_carlo_control(env, method="every_visit", num_episodes=50000)
    
    # Derive value functions from Q values
    V_first = {}
    V_every = {}
    
    for state in env.get_all_states():
        # For first-visit MC
        q_values = [Q_first[(state, a)] for a in env.actions]
        V_first[state] = max(q_values)
        
        # For every-visit MC
        q_values = [Q_every[(state, a)] for a in env.actions]
        V_every[state] = max(q_values)
    
    # Visualize policies
    grid_policy_first = visualize_policy(env, policy_first)
    grid_policy_every = visualize_policy(env, policy_every)
    
    print("\nOptimal Policy (First-Visit MC):")
    print(grid_policy_first)
    
    print("\nOptimal Policy (Every-Visit MC):")
    print(grid_policy_every)
    
    # Visualize value functions
    grid_value_first = visualize_values(env, V_first)
    grid_value_every = visualize_values(env, V_every)
    
    print("\nState Values (First-Visit MC):")
    print(np.round(grid_value_first, 2))
    
    print("\nState Values (Every-Visit MC):")
    print(np.round(grid_value_every, 2))
    
    # Compare the two methods
    print("\nDifference in State Values (First-Visit - Every-Visit):")
    print(np.round(grid_value_first - grid_value_every, 4))
    
    # Visualize the differences
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot First-Visit values
    im1 = axes[0].imshow(grid_value_first, cmap='viridis')
    axes[0].set_title('First-Visit MC Values')
    for i in range(3):
        for j in range(3):
            text = axes[0].text(j, i, f"{grid_value_first[i, j]:.2f}",
                               ha="center", va="center", color="w")
    fig.colorbar(im1, ax=axes[0])
    
    # Plot Every-Visit values
    im2 = axes[1].imshow(grid_value_every, cmap='viridis')
    axes[1].set_title('Every-Visit MC Values')
    for i in range(3):
        for j in range(3):
            text = axes[1].text(j, i, f"{grid_value_every[i, j]:.2f}",
                               ha="center", va="center", color="w")
    fig.colorbar(im2, ax=axes[1])
    
    # Plot policy (First-Visit)
    grid_for_plot = np.zeros((3, 3))
    grid_for_plot[env.start_pos] = 1
    grid_for_plot[env.high_penalty_pos] = -1
    grid_for_plot[env.goal_pos] = 2
    
    im3 = axes[2].imshow(grid_for_plot, cmap='coolwarm')
    axes[2].set_title('Optimal Policy (First-Visit MC)')
    
    # Add policy arrows
    for i in range(3):
        for j in range(3):
            state = (i, j)
            if state in [env.goal_pos, env.high_penalty_pos]:
                continue
                
            # Get best action
            q_values = [Q_first[(state, a)] for a in env.actions]
            max_q = max(q_values)
            best_action = env.actions[q_values.index(max_q)]
            
            # Draw arrow
            if best_action == "up":
                axes[2].arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='white', ec='white')
            elif best_action == "right":
                axes[2].arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
            elif best_action == "down":
                axes[2].arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='white', ec='white')
            elif best_action == "left":
                axes[2].arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
    
    # Add labels for start, goal, and high penalty
    axes[2].text(0, 0, "R", ha="center", va="center", color="k", fontweight='bold')
    axes[2].text(1, 2, "-10", ha="center", va="center", color="w")
    axes[2].text(2, 2, "+10", ha="center", va="center", color="k")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()