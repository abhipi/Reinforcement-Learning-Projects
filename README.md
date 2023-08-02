# Reinforcement Learning (RL) and Deep RL Projects <br>
* Atari (KungFu-v0): Deep Q-Learning model
* Smart-Taxi: Q-Learning
* SonicTheHedgehog: PPO (Proximal Policy Optimization) algorithm with the Actor Advantage Critic (A2C) model

This repository contains three main projects that focus on implementing various RL algorithms for different environments and games (TensorFlow, OpenAI Gym).
1. Atari (KungFu-v0) <br>
  The Atari section of the repository focuses on implementing the Deep Q-Network (DQN) algorithm for the "KungFu-v0" Atari game. The main code file for this section is DQNAtariKungFuv0.py. The steps involved in this section are as follows:
    * Pre-process the game frames.
    * Define the network architecture.
    * Train the agent using the DQN algorithm.
    * Implement utility functions for memory replay, epsilon-greedy action selection, and frame stacking.
    * Save the trained model in the saved_models directory.<br>
<img width="400" height="400" alt="image" src="https://github.com/abhipi/Reinforcement-Learning-Projects/assets/75244191/f572ae66-1908-44e8-b941-b496af52927f"><br>
2. Smart-Taxi <br>
  The Smart-Taxi section implements the Q-Learning algorithm for a Smart-Taxi environment. The main code file for this section is SmartTaxi.py. The steps involved in this section are as follows:
    * Import the necessary libraries and create an instance of the Smart-Taxi environment.
    * Initialize the Q-table with zeros to store the Q-values.
    * Define the epsilon-greedy policy function for action selection.
    * Implement the main training loop: interact with the environment, update Q-values based on the Q-Learning algorithm, and repeat for a specified number of episodes.
    * Include a testing loop to navigate the Smart-Taxi environment using the learned Q-values.
    * Evaluate the agent's performance over a specified number of test episodes and print the average score.<br>
<img width="400" height="400" alt="image" src="https://github.com/abhipi/Reinforcement-Learning-Projects/assets/75244191/eadd1887-62af-4cbf-a3af-b43d0e8ce58b"><br>
3. Sonic the Hedgehog-PPO <br>
  The Sonic the Hedgehog-PPO section focuses on training an agent to play the game "Sonic the Hedgehog" using the Proximal Policy Optimization (PPO) algorithm. The main code file for this section is run.py. The steps involved in this section are as follows:
    * Implement the Advantage Actor-Critic (A2C) architecture.
    * Train the agent using the PPO algorithm.
    * Play the game in the OpenAI Gym environment.
    * Provide utility functions for the project, such as loading model checkpoints and saving the trained model.<br>
<img width="400" height="400" src="https://github.com/abhipi/Reinforcement-Learning-Projects/assets/75244191/d09c3145-3cb3-46b9-a80e-d2a4bd9a81c5"><br>
<!-- end of the list -->
Please refer to the respective sections for more details on each algorithm's implementation and usage.
