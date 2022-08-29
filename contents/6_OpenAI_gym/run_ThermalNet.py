"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""



import gym
from tools import * # my own tool box
from RL_brain import DeepQNetwork

env = gym.make('ThermalNet-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

T_actual = [16, 14, 12, 10, 8, 6, 4, 2]

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.99,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_steps = 0
errors=[]
xs = [] # put positions of x in
x=0
for i_episode in range(100):

    observation = env.reset()
    print("resetting...")
    ep_r = 0
    training_count = 0
    while True:
        

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # x, x_dot, theta, theta_dot = observation_
        x, T_actual_i = observation_  # x stands for the position of the sensor, T_pred = [] of predicted temperature
        T_pred = get_T_pred(x,T_actual_i)
        
        error = getError(T_pred,T_actual)
        #r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        #r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5

        r1 = 1/(1+error)    # the lower the error, the better/higher the reward
        #r2 = 1/num_sensors                # the less sensors applied, the better/higher the reward
        #r3 = 0.1 if 0<x<7 else -0.1 #(env.x_threshold - abs(x))/env.x_threshold - 0.8
        reward = 10*r1 #+  r3 #'''* env.weight_of_r1''' * env.weight_of_r3 #+ env.weight_of_r2 * r2      
        #print("error:"+str(getError(T_pred,T_actual)))
        

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        env.render()

        if training_count>200:
            errors.append(error)
            xs.append(x)
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2),
                  ' error: ', round(error, 2))
            break
        training_count +=1
        observation = observation_
        total_steps += 1



RL.plot_cost()
RL.plot_error(errors)
RL.plot_positions(xs)
