import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
# Load the trained PPO model
model = PPO.load("logs/best_model.zip")
# Create the evaluation environment only for evaluate_policy
env_id = "CarRacing-v2"
# eval_envs = make_vec_env(env_id, n_envs=1)
# # Evaluate
# print("Evaluating model")
# mean_reward, std_reward = evaluate_policy(
#     model,
#     eval_envs,
#     n_eval_episodes=20,
#     deterministic=True,
# )
# print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")


# Create the environment with render_mode="human"
env = gym.make(env_id, render_mode="human")

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

