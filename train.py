import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt   

env = gym.make("Taxi-v3")

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 1.0  

episodes = 50000
max_steps = 100

rewards_per_episode = [] 

for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0  

    for step in range(max_steps):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        total_reward += reward   
        state = next_state

        if done:
            break

    rewards_per_episode.append(total_reward)   

    epsilon = max(0.01, epsilon * 0.995)

    if episode % 1000 == 0:
        print(f"Episode: {episode}")

print("Training finished!")

np.save("q_table.npy", q_table)

window = 100
smoothed = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')

plt.plot(smoothed)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Training Progress")
plt.show()

env.close()