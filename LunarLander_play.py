import gymnasium as gym
from stable_baselines3 import PPO

# Load the trained PPO model
model = PPO.load("ppo_lunarlander_gpu")

# Create the environment with render_mode="human"
env = gym.make("LunarLander-v2", render_mode="human")

# Reset the environment and extract the observation
obs, info = env.reset()

while True:
    # Use the trained model to predict actions
    action, _states = model.predict(obs, deterministic=True)

    # Take the action in the environment and unpack the result
    obs, rewards, dones, truncated, info = env.step(action)

    # Render the environment
    env.render()

    # Check if the episode is done or truncated and reset the environment
    if dones or truncated:
        obs, info = env.reset()

env.close()
