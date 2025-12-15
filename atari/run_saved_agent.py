import numpy as np
import gymnasium as gym
import ale_py
import torch
from collections import deque

gym.register_envs(ale_py)

# Import from training script
from dqn_training import DQN, preprocess_frame

# Configuration
MODEL_PATH = "saved_agents/Baseline_step_50000.pt"  # Update with your model path
N_EPISODES = 20
RENDER = True

def run_saved_agent(model_path, n_episodes=5, render=True):
    """Run a saved DQN agent."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    render_mode = "human" if render else None
    # Use full_action_space=False to get standard 4 actions (NOOP, FIRE, RIGHT, LEFT)
    # Without wrappers, the game ends after each life by default
    env = gym.make('ALE/Breakout-v5', render_mode=render_mode, full_action_space=False)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get environment info
    obs, _ = env.reset()
    n_actions = env.action_space.n
    
    # Create network
    policy_net = DQN(input_shape=(4, 80, 80), n_actions=n_actions, hidden_dim=512).to(device)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Total steps trained: {checkpoint['steps_done']}")
    print(f"Total episodes trained: {checkpoint['episodes_done']}")
    print(f"\nRunning {n_episodes} episodes...\n")
    
    # Frame stacker for state representation
    class FrameStack:
        def __init__(self, n_frames=4):
            self.n_frames = n_frames
            self.frames = deque(maxlen=n_frames)
        
        def reset(self, frame):
            processed = preprocess_frame(frame)
            for _ in range(self.n_frames):
                self.frames.append(processed)
            return np.array(self.frames)
        
        def push(self, frame):
            processed = preprocess_frame(frame)
            self.frames.append(processed)
            return np.array(self.frames)
    
    frame_stack = FrameStack(n_frames=4)
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        state = frame_stack.reset(obs)
        
        done = False
        total_reward = 0
        steps = 0
        lives = info.get('lives', 5)  # Breakout starts with 5 lives
        
        print(f"\n=== Episode {episode + 1} started with {lives} lives ===")
        
        life_lost_flag = False
        
        while not done:
            # After life loss, need to fire to restart the ball
            if life_lost_flag:
                action = 1  # FIRE action to restart
                life_lost_flag = False
            else:
                # Select action normally
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if life was lost
            new_lives = info.get('lives', lives)
            if new_lives < lives:
                print(f"  💀 Life lost at step {steps}! Lives remaining: {new_lives}")
                lives = new_lives
                life_lost_flag = True  # Need to fire on next step
            
            # Only end episode when all lives are exhausted or max timesteps reached
            # Ignore terminated flag as it triggers on each life loss
            done = (lives == 0) or truncated
            
            # Update state
            state = frame_stack.push(obs)
            
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if terminated:
            print(f"  ⚠️ Episode terminated: All lives lost or game completed")
        if truncated:
            print(f"  ⏱️ Episode truncated: Max timesteps reached")
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.1f}, Steps = {steps}, Final Lives = {lives}")
    
    env.close()
    
    # Summary statistics
    print(f"\n{'='*50}")
    print(f"Summary Statistics:")
    print(f"{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.1f}")
    print(f"Min Reward: {np.min(episode_rewards):.1f}")

if __name__ == "__main__":
    run_saved_agent(MODEL_PATH, N_EPISODES, RENDER)
