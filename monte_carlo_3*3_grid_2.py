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

# Monte Carlo Control with Q-value tracking and reward monitoring
def monte_carlo_control(env, method="first_visit", num_episodes=10000, gamma=0.9, epsilon=0.1, log_interval=100):
    # Initialize action-value function and policy
    Q = {}
    returns = {}
    policy = {}
    
    # For tracking
    q_values_history = {}
    episode_rewards = []
    episode_lengths = []
    
    # Initialize key state-action pairs to track
    key_state_actions = [
        ((0, 0), "right"),  # Starting position, moving right
        ((0, 1), "right"),  # Top middle, moving right
        ((1, 2), "down"),   # Middle right, moving down
        ((2, 1), "right")   # Bottom middle (high penalty), moving right
    ]
    
    # Initialize tracking for these key state-action pairs
    for state_action in key_state_actions:
        q_values_history[state_action] = []
    
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
        episode_reward = 0
        
        for step in range(100):  # Max steps per episode
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
            episode_reward += reward
            
            # Store state, action, reward
            episode.append((state, action, reward))
            state = next_state
            
            if done:
                break
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(len(episode))
        
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
        
        # Log Q-values for key state-action pairs
        if episode_num % log_interval == 0 or episode_num == num_episodes - 1:
            for state_action in key_state_actions:
                q_values_history[state_action].append(Q[state_action])
        
        # Update policy (greedy with respect to Q)
        for state in env.get_all_states():
            q_values = [Q[(state, a)] for a in env.actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(env.actions, q_values) if q == max_q]
            
            # Set policy to be greedy with respect to Q
            policy[state] = best_actions
    
    return Q, policy, q_values_history, episode_rewards, episode_lengths

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

# Visualize Q-values
def visualize_q_values(env, Q):
    # Define action indices for consistent ordering
    action_indices = {"up": 0, "right": 1, "down": 2, "left": 3}
    
    # Create a grid for visualization (3x3 grid, 4 actions per cell)
    q_grid = np.zeros((3, 3, 4))
    
    # Fill in grid with Q-values
    for i in range(3):
        for j in range(3):
            state = (i, j)
            for action in env.actions:
                q_grid[i, j, action_indices[action]] = Q[(state, action)]
    
    return q_grid

# Plot Q-value history and episode rewards
def plot_learning_curves(q_values_history, episode_rewards, episode_lengths, log_interval, num_episodes):
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot Q-values evolution
    episodes = np.arange(0, num_episodes, log_interval)
    if len(episodes) < len(list(q_values_history.values())[0]):
        episodes = np.append(episodes, num_episodes-1)
    
    for state_action, values in q_values_history.items():
        state, action = state_action
        label = f"State {state}, Action: {action}"
        axes[0].plot(episodes, values, label=label)
    
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Q-value')
    axes[0].set_title('Q-value Evolution for Key State-Action Pairs')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot episode rewards
    # Use moving average for smoother visualization
    window_size = 100
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    axes[1].plot(range(window_size-1, num_episodes), moving_avg)
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Episode Reward (Moving Avg)')
    axes[1].set_title(f'Episode Rewards (Moving Average, Window={window_size})')
    axes[1].grid(True)
    
    # Plot raw episode rewards with less frequency for clarity
    sample_rate = max(1, num_episodes // 1000)
    x_raw = range(0, num_episodes, sample_rate)
    y_raw = [episode_rewards[i] for i in x_raw]
    axes[2].scatter(x_raw, y_raw, alpha=0.3, s=3)
    axes[2].set_xlabel('Episodes')
    axes[2].set_ylabel('Episode Reward')
    axes[2].set_title('Raw Episode Rewards (Sampled)')
    axes[2].grid(True)
    
    plt.tight_layout()
    return fig

# Run the algorithm and visualize results
def main():
    env = GridWorld()
    
    # Initial random policy
    initial_policy = {}
    for state in env.get_all_states():
        initial_policy[state] = env.actions
    
    num_episodes = 50000
    log_interval = 100
    
    print("Running First-Visit Monte Carlo Control...")
    Q_first, policy_first, q_hist_first, rewards_first, lengths_first = monte_carlo_control(
        env, method="first_visit", num_episodes=num_episodes, log_interval=log_interval)
    
    print("Running Every-Visit Monte Carlo Control...")
    Q_every, policy_every, q_hist_every, rewards_every, lengths_every = monte_carlo_control(
        env, method="every_visit", num_episodes=num_episodes, log_interval=log_interval)
    
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
    
    # Visualize Q-values
    q_grid_first = visualize_q_values(env, Q_first)
    q_grid_every = visualize_q_values(env, Q_every)
    
    print("\nQ-Values for First-Visit MC:")
    for i in range(3):
        for j in range(3):
            state = (i, j)
            q_values = [Q_first[(state, a)] for a in env.actions]
            print(f"State {state}: {[round(q, 2) for q in q_values]} (up, right, down, left)")
    
    # Plot learning curves
    fig_first = plot_learning_curves(q_hist_first, rewards_first, lengths_first, log_interval, num_episodes)
    fig_first.suptitle('First-Visit Monte Carlo Learning Curves', fontsize=16)
    
    fig_every = plot_learning_curves(q_hist_every, rewards_every, lengths_every, log_interval, num_episodes)
    fig_every.suptitle('Every-Visit Monte Carlo Learning Curves', fontsize=16)
    
    # Compare the average reward of the last 1000 episodes
    avg_reward_first = np.mean(rewards_first[-1000:])
    avg_reward_every = np.mean(rewards_every[-1000:])
    
    print(f"\nAverage reward of last 1000 episodes:")
    print(f"First-Visit MC: {avg_reward_first:.2f}")
    print(f"Every-Visit MC: {avg_reward_every:.2f}")
    
    # Visualize the optimal path
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create a grid for visualization
    grid_for_plot = np.zeros((3, 3))
    grid_for_plot[env.start_pos] = 1
    grid_for_plot[env.high_penalty_pos] = -1
    grid_for_plot[env.goal_pos] = 2
    
    im = ax.imshow(grid_for_plot, cmap='coolwarm')
    ax.set_title('Optimal Path (First-Visit MC)')
    
    # Add optimal path arrows
    optimal_path = [
        ((0, 0), "right"),
        ((0, 1), "right"),
        ((0, 2), "down"),
        ((1, 2), "down")
    ]
    
    for state, action in optimal_path:
        i, j = state
        if action == "up":
            ax.arrow(j, i, 0, -0.3, head_width=0.1, head_length=0.1, fc='white', ec='white')
        elif action == "right":
            ax.arrow(j, i, 0.3, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
        elif action == "down":
            ax.arrow(j, i, 0, 0.3, head_width=0.1, head_length=0.1, fc='white', ec='white')
        elif action == "left":
            ax.arrow(j, i, -0.3, 0, head_width=0.1, head_length=0.1, fc='white', ec='white')
    
    # Add state values from First-Visit MC
    for i in range(3):
        for j in range(3):
            state = (i, j)
            ax.text(j, i, f"V={V_first[state]:.2f}", ha="center", va="center", 
                    color="w" if (i, j) in [env.high_penalty_pos, env.goal_pos] else "k",
                    fontsize=9)
    
    # Add labels for start, goal, and high penalty
    ax.text(0, 0, "Start", ha="center", va="bottom", color="k", fontweight='bold', fontsize=10)
    ax.text(1, 2, "-10", ha="center", va="bottom", color="w", fontweight='bold', fontsize=10)
    ax.text(2, 2, "+10", ha="center", va="bottom", color="k", fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()