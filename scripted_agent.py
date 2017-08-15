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
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


# clean up unused imports
# clean up unused hyper parameters in different files
# comment and clean up code

# make changes to below code

# additionally there isn't 3 actions, we also need to define x,y position for screen clicking


# lines that need to change
# 191, 192
# most of work()
# then how does this integrate with the bot



# change optimizer to shared RMS prop (I don't know if this is even possible in TF)
# fix the process frames


# TODO

# reward function needs to either be based on win, tie or loss (1, 0 or -1)
# Dont use grayscale




class MoveToBeacon(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""
  def step(self, obs,takeAction,actionParameters):
    super(MoveToBeacon, self).step(obs)
    return actions.FunctionCall(takeAction,actionParameters)
