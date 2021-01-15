import numpy as np
import gym
import retro_contest
from retro_contest.local import make
from retro import make as make_retro

from baselines.common.atari_wrappers import FrameStack
import cv2
cv2.ocl.setUseOpenCL(False)

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96  #Resizing the frame to 96*96*1 ie.grayscale
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame


class ActionsDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []

        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): 
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    
    def reward(self, reward):

        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): 
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): 
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def make_env(env_idx):
    
    dicts = [
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'SpringYardZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'GreenHillZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'StarLightZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'MarbleZone.Act3'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'ScrapBrainZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act2'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act1'},
        {'game': 'SonicTheHedgehog-Genesis', 'state': 'LabyrinthZone.Act3'}
        

    ]
    
    print(dicts[env_idx]['game'], dicts[env_idx]['state'], flush=True) #Create the environment ie. the desired level
    
    env = make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'], bk2dir="./records")#record='/tmp')

    # Build the actions array, 
    env = ActionsDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

  
    env = PreprocessFrame(env)

 
    env = FrameStack(env, 4)

    
    env = AllowBacktracking(env) #To allow backtracking if further exploration is not possible



    return env



def make_train_0():
    return make_env(0)

def make_train_1():
    return make_env(1)

def make_train_2():
    return make_env(2)

def make_train_3():
    return make_env(3)

def make_train_4():
    return make_env(4)

def make_train_5():
    return make_env(5)

def make_train_6():
    return make_env(6)

def make_train_7():
    return make_env(7)

def make_train_8():
    return make_env(8)

def make_train_9():
    return make_env(9)

def make_train_10():
    return make_env(10)

def make_train_11():
    return make_env(11)

def make_train_12():
    return make_env(12)

def make_test_level_Green():
    return make_test()


def make_test():
    env = make_retro(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2', record="./records") 
    env = ActionsDiscretizer(env)
    env = RewardScaler(env)
    env = PreprocessFrame(env)

    env = FrameStack(env, 4)

    env = AllowBacktracking(env)

    return env
