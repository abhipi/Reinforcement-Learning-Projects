import retro

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record=True)
env.reset()
time = 0
total_reward = 0
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    time += 1
    if time % 10 == 0:
        if info:
            info_content = {key: value for key, value in info.items()}
            print(f'time={time}, info: {info_content}')
        env.render()
    total_reward += reward
    if reward > 0:
        print(f'time: {time}, reward: {reward}, current_reward: {total_reward}')
    if reward < 0:
        print(f'time: {time}, penalty: {reward}, current_reward: {total_reward}')
    if done:
        # This happens both, when time and lives are up, and when game is completed
        env.render()
        print('done!')
        break
print(f'time: {time}, total_reward: {total_reward}')
