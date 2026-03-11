from stable_baselines3 import DQN
from traffic_env import TrafficLightEnv
import numpy as np

# Create environment
env = TrafficLightEnv()

# Create DQN agent
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000, batch_size=32)

# Train the agent
print("Training RL agent...")
model.learn(total_timesteps=100000)

# Save the trained model
model.save("traffic_light_rl")
print("Model saved as traffic_light_rl.zip")

# Test the trained agent
print("Testing trained agent...")
obs, _ = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    print(f"Reward: {reward}")
    if done or truncated:
        obs, _ = env.reset()
