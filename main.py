import gymnasium as gym
import random
env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
position_bin = round(observation[0], 1)
velocity_bin = round(observation[1])
angle_bin = int(observation[2] * 10)
angular_velocity_bin = round(observation[3])

current_state_key = (position_bin, velocity_bin, angle_bin, angular_velocity_bin)

table = {}

print("Starting")

action = env.action_space.sample()

for step in range(1000):

     observation, reward, terminated, truncated, info = env.step(action)

     position_bin = round(observation[0], 1)
     velocity_bin = round(observation[1])
     angle_bin = int(observation[2] * 10)
     angular_velocity_bin = round(observation[3])

     state_key = (position_bin, velocity_bin, angle_bin, angular_velocity_bin)
     next_state_key = state_key

     if current_state_key not in table:
          table[current_state_key] = [0.0, 0.0]

     next_score = table.get(next_state_key, [0.0, 0.0])
     y = 0.95
     a = 0.1
     old_q = table[current_state_key][action]
     target = reward + (y * max(next_score))
     new_q = old_q + a * (target - old_q)

     if action == 1:
          table[current_state_key][1] = new_q
     else:
          table[current_state_key][0] = new_q

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

     current_state_key = next_state_key
     if terminated or truncated:
          print("Episode finished")
          action = env.action_space.sample()
          observation, info = env.reset()
          position_bin = round(observation[0], 1)
          velocity_bin = round(observation[1])
          angle_bin = int(observation[2] * 10)
          angular_velocity_bin = round(observation[3])

          current_state_key = (position_bin, velocity_bin, angle_bin, angular_velocity_bin)
        

print(table)
env.close()
print("Finished")