import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch
import argparse
# Use argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v2")
parser.add_argument("--timesteps", type=int, default=int(5e5), help="Total timesteps for training")
args = parser.parse_args()

print("torch.cuda.is_available()", torch.cuda.is_available())  # Should return True

# Create the LunarLander environment
env_id = "CarRacing-v2"
n_envs = 16
env = make_vec_env(env_id, n_envs=n_envs)

# Create the evaluation envs
eval_envs = make_vec_env(env_id, n_envs=5)

# Adjust evaluation interval depending on the number of envs
eval_freq = int(1e5)
eval_freq = max(eval_freq // n_envs, 1)

# Create evaluation callback to save best model
# and monitor agent performance
eval_callback = EvalCallback(
    eval_envs,
    best_model_save_path="./logs/",
    eval_freq=eval_freq,
    n_eval_episodes=10,
)
# Instantiate the agent
# Hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    gae_lambda=0.98,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    verbose=1,
    device='cuda',
)

# Train the agent (you can kill it before using ctrl+c)
try:
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
except KeyboardInterrupt:
    pass

# Save the trained model
# model.save("/logs/ppo_lunarlander_gpu")

# Close the environment
env.close()

print("Training complete and model saved as 'logs/best_model.zip' after {args.timesteps} timesteps.")
