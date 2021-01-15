import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import gym
import math
import os

import acmodels as model
import a2c as policies
import env
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)):
        model.learn(policy=policies.A2CPolicy,
                            env=SubprocVecEnv([env.make_train_0, env.make_train_1, env.make_train_2, env.make_train_3, env.make_train_4, env.make_train_5,env.make_train_6,env.make_train_7,env.make_train_8,env.make_train_9,env.make_train_10,env.make_train_11,env.make_train_12 ]), 
                            nsteps=2048, 
                            total_timesteps=10000000,
                            gamma=0.99,
                            lam = 0.95,
                            vf_coef=0.5,
                            ent_coef=0.01,
                            lr = 2e-4,
                            max_grad_norm = 0.5, 
                            log_interval = 10
                            )

if __name__ == '__main__':
    main()
