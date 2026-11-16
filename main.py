import gymnasium as gym
import random
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

table = {}

print("Starting")

action = env.action_space.sample()

for step in range(500):
    observation, reward, terminated, truncated, info = env.step(action)

    position_bin = round(observation[0], 1)
    velocity_bin = round(observation[1])
    angle_bin = int(observation[2] * 10)
    angular_velocity_bin = round(observation[3])

    state_key = (position_bin, velocity_bin, angle_bin, angular_velocity_bin)

    if state_key not in table:
         table[state_key] = [0.0, 0.0]

    if action == 1:
         table[state_key][1] = reward
    else:
         table[state_key][0] = reward

    scores = table.get(state_key, [0.0, 0.0])
    roll = random.randint(1, 10)
    chance = True if roll == 10 else False

    if not chance: # Exploitation
        if scores[0] > scores[1]:
            action = 0
        else:
            action = 1
    else: # Exploration
         action = random.randint(0, 1)


    if terminated or truncated:
        print("Episode finished")
        action = env.action_space.sample()
        observation, info = env.reset()

print(table)
env.close()
print("Finished")