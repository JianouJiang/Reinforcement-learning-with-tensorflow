"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import copy
import statistics
import gym
from tools import * # my own tool box
import matplotlib.pyplot as plt
from RL_brain import DeepQNetwork

env = gym.make('ThermalNet-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

#T_actual = [ 4, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 14]
x = [i for i in range(0,41)]
T_actual = sinT(x)
'''
RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.99,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)
'''
env.action_space.n=int(3*3) # 9 actions for 3 agents in one agent...; 3 is for three individual agents
nb_agents=1
total_nb_action = 9 # for 3 agents
nb_action = 3 # for each agent
nb_states = 2 # x and T for each agent
n_features = int(3*nb_states) # =3 agents * 2 states = x1,T1,x2,T2,x3,T3; 6 features in obs
RL = DeepQNetwork(n_actions=env.action_space.n,
                  nb_agents=nb_agents,
                  n_features=n_features,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
errors=[]

xs = [ [] for i in range(nb_agents)] # put positions of x in

x=0

for i_episode in range(500):
    
    agents = env.reset()
    
    observations = [agents[i]["state"] for i in range(env.getNb_Agents())] # [( (x1,T1),(x2,T2),(x3,T3) ), ( (x1,T1),(x2,T2),(x3,T3) ), ( (x1,T1),(x2,T2),(x3,T3)  )] # this is the final structure, TODO!
    #print("after reset")
    #print(observations)
    observations_old = copy.copy(observations)
    ep_r = 0
    training_count = 0
    while True:

        actions = RL.choose_action(observations)
        #print("after actions")
        #print(observations)
        observations_, rewards, done, info, T_pred = env.step(actions) # this rewards evaluate each sensor's prediction (based on a small window)
        #print("after step")
        #print(observations_old)
        #stdev_rewards = statistics.stdev(rewards)
       
        error = getError(T_pred,T_actual)
        #print(error)
        
        r1 = 5/(1e-6+error)    # the lower the error, the better/higher the reward
        if error>8:
            r1=-1
            rewards -=rewards
            
        elif error<2.5:
            r1=r1*10
        #r2 = 1/(stdev_rewards+1)
        reward =  r1 #+  r3 #'''* env.weight_of_r1''' * env.weight_of_r3 #+ env.weight_of_r2 * r2      
                
        rewards += reward # TODO! Fix grammer, add overall reward to individual reward: so an agent's reward is upto the individual's performace as well as the overal performace
        
        
        #print("...testing...")
        #print(observations)
        #print(observations_)
        RL.store_transition(observations, actions, rewards, observations_)

        ep_r += sum(rewards)/len(rewards)
        if total_steps > 4 and total_steps % 5==0:
            RL.learn()

        #env.render()
        
        if training_count>200:
            print(rewards)
            env.Myrender(T_pred, error, i_episode, observations_)
            errors.append(error)
            agents = env.getAgentsStates()    
            
            for i in range(len(agents)):
                state_i = agents[i]["state"]    # [x1,T1,x2,T2,x3,T3]
                
                xis = [ [] for j in range(int(total_nb_action/nb_action))] # put positions of x in
                for j in range(int(total_nb_action/nb_action)): #total_nb_action/nb_action = 9/3 = 3 agents
                    xi = state_i[int(j*nb_states)]
                    T_actual_i = state_i[int(j*nb_states+1)]
                    xis[j].append(xi)
                xs[i].append(xis) # 
                
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2),
                  ' error: ', round(error, 2))
            break
        
        training_count +=1
        observations = observations_
        total_steps += 1


RL.plot_cost()
RL.plot_error(errors)
RL.plot_positions(xs)
