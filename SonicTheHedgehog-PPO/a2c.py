import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from baselines.common.distributions import make_pdtype #probability distribution over actions

def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=tf.nn.relu,
                            kernel_initializer=tf.orthogonal_initializer(gain=gain))



def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))


class A2CPolicy(object): #this class defines the advantage actor critic architechture
    def __init__(self, sess, ob_space, action_space, nbatch, nsteps, reuse = False):
        
        gain = np.sqrt(2)

        
        self.pdtype = make_pdtype(action_space)

        height, weight, channel = ob_space.shape
        ob_shape = (height, weight, channel)

        
        inputs_ = tf.placeholder(tf.float32, [None, *ob_shape], name="input")

        
        scaled_images = tf.cast(inputs_, tf.float32) / 255.

        with tf.variable_scope("acmodels", reuse = reuse):
            conv1 = conv_layer(scaled_images, 32, 8, 4, gain)
            conv2 = conv_layer(conv1, 64, 4, 2, gain)
            conv3 = conv_layer(conv2, 64, 3, 1, gain)
            flatten1 = tf.layers.flatten(conv3)
            fc_common = fc_layer(flatten1, 512, gain=gain)

            self.pd, self.pi = self.pdtype.pdfromlatent(fc_common, init_scale=0.01)

            
            vf = fc_layer(fc_common, 1, activation_fn=None)[:, 0]

        self.initial_state = None
        a0 = self.pd.sample()

        # Function use to take a step returns action to take and V(s)
        def step(state_in, *_args, **_kwargs):
            action, value = sess.run([a0, vf], {inputs_: state_in})
           
                      
            return action, value

        # Function that calculates only the V(s)
        def value(state_in, *_args, **_kwargs):
            return sess.run(vf, {inputs_: state_in})

        # Function that output only the action to take
        def select_action(state_in, *_args, **_kwargs):
            return sess.run(a0, {inputs_: state_in})

        self.inputs_ = inputs_
        self.vf = vf
        self.step = step
        self.value = value
        self.select_action = select_action
