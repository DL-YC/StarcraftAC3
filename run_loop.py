# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import os

from random import choice
from time import sleep

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import acNetwork

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]




# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Crops and resizes image as necessary
def process_frame(frame):
    s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(s,[84,84])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0


    s = frame[49:193, 8:152] # Remove borders and counters
    s = np.mean(s, axis=2) # convert to grayscale
    s[s != 0] = 1
   
    return s

# Discounts gamma from future rewards
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker():
    def __init__(self,name,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = acNetwork.AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)        
        
        #self.actions = [1, 4, 5] # Stay, Move Left, Move Right
        #self.env = gym.make('Breakout-v0')
        
    def train(self, global_AC, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages}
        v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,agents,env,max_episode_length,gamma,global_AC,sess,coord,saver,max_frames=0):
        total_frames = 0

        action_spec = env.action_spec()
        observation_spec = env.observation_spec()
        for agent in agents:
          agent.setup(observation_spec, action_spec)

        try:
          while True:
            timesteps = env.reset()
            for a in agents:
              a.reset()
            while True:
              total_frames += 1
              actions = [agent.step(timestep,_NO_OP, [])
                         for agent, timestep in zip(agents, timesteps)]

              if max_frames and total_frames >= max_frames:
                return
              if timesteps[0].last():
                break
              timesteps = env.step(actions)
        except KeyboardInterrupt:
          pass
  
        




        # episode_count = sess.run(self.global_episodes)
        # total_steps = 0
        # print("Starting worker %d" % self.number)
        # with sess.as_default(), sess.graph.as_default():                 
        #     while not coord.should_stop():
        #         sess.run(self.update_local_ops)
        #         episode_buffer = []
        #         episode_values = []
        #         episode_frames = []
        #         episode_reward = 0
        #         episode_step_count = 0
        #         d = False
                
        #         s = self.env.reset()
        #         episode_frames.append(s)
        #         s = process_frame(s)
                
        #         while d == False:
        #             #Take an action using probabilities from policy network output.
        #             a_dist,v = sess.run(
        #                 [self.local_AC.policy, self.local_AC.value], 
        #                 feed_dict={
        #                     self.local_AC.inputs: [s]})
        #             a = np.random.choice(a_dist[0], p=a_dist[0])
        #             a = np.argmax(a_dist == a)

        #             s1, r, d, _ = self.env.step(self.actions[a])
        #             if d == False:
        #                 episode_frames.append(s1)
        #                 s1 = process_frame(s1)
        #             else:
        #                 s1 = s
                        
        #             episode_buffer.append([s,a,r,s1,d,v[0,0]])
        #             episode_values.append(v[0,0])

        #             episode_reward += r
        #             s = s1                    
        #             total_steps += 1
        #             episode_step_count += 1
                    
        #             # If the episode hasn't ended, but the experience buffer is full, then we
        #             # make an update step using that experience rollout.
        #             if len(episode_buffer) == 40 and d != True and episode_step_count != max_episode_length - 1:
        #                 # Since we don't know what the true final return is, we "bootstrap" from our current
        #                 # value estimation.
        #                 v1 = sess.run(self.local_AC.value, 
        #                     feed_dict={
        #                         self.local_AC.inputs: [s]})[0,0]
        #                 v_l,p_l,e_l,g_n,v_n = self.train(global_AC, episode_buffer, sess, gamma, v1)
        #                 episode_buffer = []
        #                 sess.run(self.update_local_ops)
        #             if d == True:
        #                 break
                                            
        #         self.episode_rewards.append(episode_reward)
        #         self.episode_lengths.append(episode_step_count)
        #         self.episode_mean_values.append(np.mean(episode_values))
                
        #         # Update the network using the experience buffer at the end of the episode.
        #         if len(episode_buffer) != 0:
        #             v_l,p_l,e_l,g_n,v_n = self.train(global_AC,episode_buffer,sess,gamma,0.0)
                                
                    
        #         # Periodically save model parameters and summary statistics.
        #         if episode_count % 5 == 0 and episode_count != 0:
        #             if episode_count % 50 == 0 and self.name == 'worker_0':
        #                 saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
        #                 print("Saved Model")

        #             mean_reward = np.mean(self.episode_rewards[-5:])
        #             mean_length = np.mean(self.episode_lengths[-5:])
        #             mean_value = np.mean(self.episode_mean_values[-5:])
        #             summary = tf.Summary()
        #             summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
        #             summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
        #             summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
        #             summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
        #             summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
        #             summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
        #             summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
        #             summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
        #             self.summary_writer.add_summary(summary, episode_count)

        #             self.summary_writer.flush()
        #         if self.name == 'worker_0':
        #             sess.run(self.increment)
        #         episode_count += 1



