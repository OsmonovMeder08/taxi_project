import gymnasium as gym
import numpy as np
import time

env = gym.make("Taxi-v3", render_mode="human")

q_table = np.load("q_table.npy")

max_steps = 100

for episode in range(10):
    state = env.reset()[0]
    total_reward = 0

    print(f"\n=== Episode {episode + 1} ===")

    for step in range(max_steps):
        action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        print(f"Step reward: {reward}")

        state = next_state
        time.sleep(0.3)

        if done:
            break

    print(f"Total reward: {total_reward}")
    time.sleep(1)

env.close()