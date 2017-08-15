#!/usr/bin/python
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
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import threading

from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import stopwatch
from pysc2.env import acNetwork

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

FLAGS = flags.FLAGS
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 84,
                     "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64,
                     "Resolution for minimap feature layers.")

flags.DEFINE_integer("max_agent_steps", 2500, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", 0, "Game steps per episode.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_string("agent", "pysc2.agents.a3cAgent.py",
                    "Which agent to run")

flags.DEFINE_enum("agent_race", None, sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.difficulties.keys(),
                  "Bot's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", None, "Name of a map to use.")
flags.mark_flag_as_required("map")


# Define hyperparameters and constants
s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1

# sc2 api agent defaults to do 2500 steps before quitting
max_episode_length = 2500

gamma = .99 # discount rate for advantage estimation and reward discounting
a_size = 3 # Agent can do NO_OP, MOVE_SCREEN, or SELECT_ARMY


# If you want to import a pretrained model or pick up where you left off, set load_model to true
# and model_path to the location
load_model = False
model_path = './model'


def run_thread(currentWorker, max_episode_length,gamma,master_network,sess,coord,saver, agent_cls, map_name, visualize):
  with sc2_env.SC2Env(
      map_name,
      agent_race=FLAGS.agent_race,
      bot_race=FLAGS.bot_race,
      difficulty=FLAGS.difficulty,
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
      minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    agent = agent_cls()
    currentWorker.work([agent], env, max_episode_length, gamma, master_network, sess, coord, saver, FLAGS.max_agent_steps)
    if FLAGS.save_replay:
      env.save_replay(agent_cls.__name__)

def _main(unused_argv):
  """Run an agent."""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)


  # my code
  tf.reset_default_graph()

  if not os.path.exists(model_path):
      os.makedirs(model_path)

  global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
  trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
  master_network = acNetwork.AC_Network(s_size,a_size,'global',None) # Generate global network

  workers = []
  # Create worker classes
  for i in range(FLAGS.parallel):
    workers.append(run_loop.Worker(i,s_size,a_size,trainer,model_path,global_episodes))
  saver = tf.train.Saver(max_to_keep=5)
    
  with tf.Session() as sess:
      coord = tf.train.Coordinator()
      if load_model == True:
          print('Loading Model...')
          ckpt = tf.train.get_checkpoint_state(model_path)
          saver.restore(sess,ckpt.model_checkpoint_path)
      else:
          sess.run(tf.global_variables_initializer())
          
      #This is where the asynchronous magic happens.
      #Start the "work" process for each worker in a separate threat.
      worker_threads = []
      for worker in workers:
          t = threading.Thread(target=run_thread, args=(
              worker, max_episode_length,gamma,master_network,sess,coord,saver, agent_cls, FLAGS.map, FLAGS.render and i == 0))
          worker_threads.append(t)
          t.start()
      coord.join(worker_threads)

      if FLAGS.profile:
        print(stopwatch.sw)


def main():  # Needed so setup.py scripts work.
  app.really_start(_main)


if __name__ == "__main__":
  main()





