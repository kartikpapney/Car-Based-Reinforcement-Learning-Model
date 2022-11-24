# Aritificial Intelligence
# CO401/ CS525
# A Reinforcement Learning based self learning Model in Mountain Car Environment


import gym
import numpy as np

# Learning Environment

# https://www.gymlibrary.dev/environments/classic_control/mountain_car/
# Deterministic actions
# 0: Accelerate to the left
# 1: Don't accelerate
# 2: Accelerate to the right

# The goal is to reach the flag placed on top of the right hill as quickly as possible, 
# as such the agent is penalised with a reward of -1 for each timestep.
env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95 
REWARD = 0


EPISODES = 20000 
SHOW_EVERY = 50 

# Total Count of Episodes
# Screen render every 100th episode

# 1. DISCRETE_OS_SIZE = We can varry it according to the number of different discrete state we want 
# 2. env.observation_space.high = position of the car along the x-axis
# 3. env.observation_space.low = velocity of the car

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE



# epsilon helps us in selecting choice between
# 1. action with max q value in the table
# 2. a random action from the table
# the value keeps on decrementing with the episodes 
# Will keep on decrementing till the END_EPSILON_DECAYING episode


epsilon = 1 
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table

for episode in range(EPISODES):

    # env.reset() resets the position and velocity of car
    # 1. position: random in range [-1.2, 0.6] 
    # 2. velocity: random in range [-0.07, 0.07]
    # 2. Converting position and valocity to discrete value in range [0, 20]
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(f"{episode} Done!!!")
    else:
        render = False

    # Loop will run till we get the goal state
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, truncated = env.step(action)

        # Descrete value between 0-20 based on the value of new_state
        new_discrete_state = get_discrete_state(new_state)

        # Render screen every SHOW_EVERYth episode
        if render:
            env.render()

        if not done:

            # Get the Maximum possible Q value for the future state
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q-current_q)
            # new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with Reward
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = REWARD

        discrete_state = new_discrete_state

    # Will keep on decrementing epsilon till the END_EPSILON_DECAYING episode.
    # After that probability of selecting an action with max q value or selecting a random action will be same
    if END_EPSILON_DECAYING > episode:
        epsilon -= epsilon_decay_value

# Close the environment
env.close()