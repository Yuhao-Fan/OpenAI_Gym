import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
print("torch.cuda.is_available()", torch.cuda.is_available())  # Should return True

# Create the LunarLander environment
env = make_vec_env("LunarLander-v2", n_envs=1)

# Create the PPO model, specifying the device to 'cuda' for GPU usage
model = PPO("MlpPolicy", env, verbose=1, device='cuda')  # Use 'cuda' for GPU

# Train the model for 100,000 timesteps
model.learn(total_timesteps=100000)

# Save the trained model
model.save("/models/ppo_lunarlander_gpu")

# Close the environment
env.close()

print("Training complete and model saved as 'ppo_lunarlander_gpu'")
