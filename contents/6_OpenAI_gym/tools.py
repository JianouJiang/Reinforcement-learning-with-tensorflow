import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import tensorflow as tf

# read from .txt file
x = [i for i in range(0,41)]
def sinT(x):
    x = np.array(x)
    T = 10*(1+np.sin(2*np.pi/len(x) * x))+2
    return T.tolist()

T_actual = sinT(x) # [ 4, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 14]
len_of_T = len(T_actual)
def getError(T_pred, T_actual):
    total_Error = 0
    for i in range(len(T_pred)):
        total_Error += abs(T_pred[i]-T_actual[i]) # TODO, abviously, the abs should be here, but only modified for testing reasons
    return total_Error/len(T_pred)
    
def getLocalError1D(T_pred, T_actual, xi):
    window_radius = 1
    local_Error = 0
    for j in range(int(xi)-window_radius,int(xi)+window_radius+1):
        local_Error += abs(T_pred[j]-T_actual[j])
    return local_Error/ (1+2*window_radius)

def get_T_pred(x,T_actual_i): # TODO! a simplified method for now, real method should use heat diffusion equation
    T_pred = []
    for i in range(len_of_T):
        T_pred.append(T_actual_i)
    return T_pred
    
def is_at_sensors(observations_, i):
    bolean = False
    for observation in observations_:
        #print(observation)
        xi, Ti = observation
        if xi==i:
            bolean = True
    return bolean    

def addSensorTemperature(observations_, T_pred):
    for observation in observations_:
        xi, Ti = observation
        T_pred[int(xi)] = Ti

    return T_pred

def not_steady_state(T_pred):
    error_threshold = 0.001
    total_Error = 0
    for i in range(1,len(T_pred)-1): 
        total_Error +=  abs(- 2*T_pred[i] + T_pred[i-1] + T_pred[i+1])
    if total_Error/len(T_pred)<error_threshold:
        #print("converged")
        return False
    else:
        #print(total_Error/len(T_pred))
        return True
    
    
# Heat diffusion equation with periodic B.C and Dirichlet temperatures from sensors    
def get_T_pred_MultiSensors(observations_): 
    
    # get the range of x coorindate, x is also the index of the coordinate
    x_min, x_max = 0, len(T_actual)#getXRange() 
    avg_T = sum(T_actual)/len(T_actual)
    # thermal diffusivity, since we are running to steady-state, we dont really care about the lambda
    lambda_i = 500 
    # low cfl number for a stable convergence 
    cfl = 0.95
    # coarse grid size
    dx = 1
    # stable time step
    dt = cfl * 1/(2*lambda_i)*dx*dx 
    # initialise T_pred and T_pred_new with 0s
    T_pred = [avg_T]*x_max
    # append sensor temperatures to the right locations
    T_pred = addSensorTemperature(observations_, T_pred)
    # store the temperature of the next time step
    T_pred_new = [avg_T]*x_max
    #print(T_pred) 
    cnt = 0
    while not_steady_state(T_pred) and cnt<1000:
        cnt +=1
        for i in range(x_max):
            # at the boundaries
            if i==x_min or i==(x_max-1):
                T_pred_new[i] = T_pred[i]
            # inside the domain
            else: 
                # at the thermal-couples' locations
                if is_at_sensors(observations_, i):
                    T_pred_new[i] = T_pred[i]
                else:
                    gamma_i = lambda_i * dt/(dx*dx)
                    T_pred_new[i] = gamma_i * (- 2*T_pred[i] + T_pred[i-1] + T_pred[i+1]) + T_pred[i]
        
        # put the next time step temperature to the old time step temperature
        T_pred = T_pred_new
        # apply periodic B.C
        # at the left B.C.
        T_pred[x_min] = T_pred_new[-2] 
        # at the right B.C.
        T_pred[int(x_max-1)] = T_pred_new[int(x_min+1)]
        #print(T_pred) 
        
    return T_pred


