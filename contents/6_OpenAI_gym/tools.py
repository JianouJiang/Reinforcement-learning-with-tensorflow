import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import tensorflow as tf


# read from .txt file

T_actual = [16, 14, 12, 10, 8, 6, 4, 2]
len_of_T = len(T_actual)
def getError(T_pred, T_actual):
  total_Error = 0
  for i in range(len(T_pred)):
    total_Error += abs(T_pred[i]-T_actual[i])
  return total_Error/len(T_pred)

def get_T_pred(x,T_actual_i): # TODO! a simplified method for now, real method should use heat diffusion equation
  T_pred = []
  for i in range(len_of_T):
    T_pred.append(T_actual_i)
  return T_pred

def get_T_pred_2Sensors(x1,x2,T_actual_1,T_actual_2): # TODO! a simplified method for now, real method should use heat diffusion equation
  T_pred = []
  T_actual_i = 0.5* (T_actual_1+T_actual_2)
  for i in range(len_of_T):
    T_pred.append(T_actual_i)
  return T_pred