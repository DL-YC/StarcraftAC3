from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from pysc2.lib import app
import gflags as flags

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os

from random import choice
from time import sleep
from time import time



# Define hyperparameters and constants
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1

# sc2 api agent defaults to do 2500 steps before quitting
max_episode_length = 2500

gamma = .99 # discount rate for advantage estimation and reward discounting
a_size = 3 # Agent can do NO_OP, MOVE_SCREEN, or SELECT_ARMY






## Below defines the global network
# This has the definition for the neural network which takes
# screen -> policy and value function approximation
# This network is updated by changes from the individual
# workers/agents running simultaneously

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            
            # This bot only uses the screen input, it does not use the minimap or any feature vectors

            # The following defines the neural network used to determine the policy and value function
            #Input and visual encoding layers

            # Reshape input image to correct size
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])

            # Two convolutional layers as described in the paper
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')

            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')

            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.relu)
            
            # Do I need more hidden layers? They say "used as input to linear layers"

            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(hidden, a_size,
                                                activation_fn=tf.nn.softmax,
                                                weights_initializer=normalized_columns_initializer(0.01),
                                                biases_initializer=None)
            self.value = slim.fully_connected(hidden, 1,
                                                activation_fn=None,
                                                weights_initializer=normalized_columns_initializer(1.0),
                                                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-6))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))



