import gym
import numpy as np
import random
import time
    
env=gym.make("Taxi-v3")
env.render()
state_space=env.observation_space.n
print("There are ", state_space,"posiible states")
action_space=env.action_space.n
print("There are ", action_space, "possible actions") #to determine the dimensions of the   Q-Table
Q=np.zeros((state_space,action_space))
print(Q)
print(Q.shape)
total_episodes=25000
total_test_episodes=100
max_steps=200

learning_rate=0.01
gamma=0.99
epsilon=1.0
max_epsilon=1.0
min_epsilon=0.001
decay_rate=0.01

def epsilon_greedy_policy(Q,state):
    if(random.uniform(0,1)>epsilon):
        action=np.argmax(Q[state])
    else:
        action=env.action_space.sample()

    return action

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    
    for step in range(max_steps):
        action = epsilon_greedy_policy(Q, state)
        new_state, reward, done, info = env.step(action)
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * 
                                    np.max(Q[new_state]) - Q[state][action])      
        
        if done == True: 
            break
        
        
        state = new_state

rewards = []
frames = []
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)
    for step in range(max_steps):
        env.render()     
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q[state][:])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        
        if done:
            rewards.append(total_rewards)
            #print ("Score", total_rewards)
            break
        state = new_state
env.close()
print ("Score over time: " +  str(sum(rewards)/total_test_episodes))
