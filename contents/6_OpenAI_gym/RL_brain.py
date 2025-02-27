"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            nb_agents,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.nb_agents = nb_agents
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        #print("memory size:")
        #print(self.memory_size)
        #print("n_features * 2 + 2")
        #print(n_features * 2 + 2)
        
        #print(np.zeros((self.memory_size, nb_agents , n_features * 2 + 2)))
        
        self.agents = [{}] * self.nb_agents
        
        # initiate memory and build net for each and every agent
        for i in range(self.nb_agents):
            # initiate memory
            self.agents[i].update({"memory": np.zeros((self.memory_size, n_features * 2 + 2))})
            # initiate build_net related parameters
            self.agents[i].update({"s": None})
            self.agents[i].update({"q_target": None})
            self.agents[i].update({"q_eval": None})
            self.agents[i].update({"loss": None})
            self.agents[i].update({"_train_op": None})
            self.agents[i].update({"s_": None})
            self.agents[i].update({"q_next": None})
            self.agents[i].update({"cost": None})
            # consist of [target_net, evaluate_net]
            self._build_net(i)
            
            t_params = tf.get_collection('target_net_params'+str(i))
            e_params = tf.get_collection('eval_net_params'+str(i))
            #self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
            self.agents[i].update({"replace_target_op": [tf.assign(t, e) for t, e in zip(t_params, e_params)]})


        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
    
    def getAgentsMemory(self):
        return self.agents
        
    def getNb_Agents(self):
        return self.nb_agents 
    
    def _build_net(self,i):
        # ------------------ build evaluate_net ------------------
        self.agents[i]["s"] = tf.placeholder(tf.float32, [None, self.n_features], name='s'+str(i))  # input; self.si = tf.placeholder(tf.float32, [None, self.n_features], name='si')  # input
        self.agents[i]["q_target"] = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target'+str(i))  # for calculating loss; self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'+str(i)):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params'+str(i), tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'+str(i)):
                w1 = tf.get_variable('w1'+str(i), [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1'+str(i), [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.agents[i]["s"], w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'+str(i)):
                w2 = tf.get_variable('w2'+str(i), [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2'+str(i), [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.agents[i]["q_eval"] = tf.matmul(l1, w2) + b2   # self.q_eval = tf.matmul(l1, w2) + b2
                

        with tf.variable_scope('loss'+str(i)):
            #self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            self.agents[i]["loss"] = tf.reduce_mean(tf.squared_difference(self.agents[i]["q_target"], self.agents[i]["q_eval"]))
        with tf.variable_scope('train'+str(i)):
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self.agents[i]["_train_op"] = tf.train.RMSPropOptimizer(self.lr).minimize(self.agents[i]["loss"])

        # ------------------ build target_net ------------------
        #self.si_ = tf.placeholder(tf.float32, [None, self.n_features], name='si_')    # input
        self.agents[i]["s_"] = tf.placeholder(tf.float32, [None, self.n_features], name='s_'+str(1))    # input
        with tf.variable_scope('target_net'+str(i)):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params'+str(i), tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'+str(i)):
                w1 = tf.get_variable('w1'+str(i), [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1'+str(i), [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.agents[i]["s_"], w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'+str(i)):
                w2 = tf.get_variable('w2'+str(i), [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2'+str(i), [1, self.n_actions], initializer=b_initializer, collections=c_names)
                #self.q_next = tf.matmul(l1, w2) + b2
                self.agents[i]["q_next"] = tf.matmul(l1, w2) + b2
    
        '''
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
        '''
    def store_transition(self, s, a, r, s_): #(observations, actions, rewards, observations_) TODO: defo wrong here...need to study for multi-agent scenarios
        
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        # when there are multiple agents, each has (x,T)
        
        for i in range(self.nb_agents):
            observation, action, reward, observation_ = s[i], a[i], r[i], s_[i]
            transition = np.hstack((observation, [action, reward], observation_))
            self.agents[i]["memory"][index, :] = transition
        
        # when there is only 1 agent, but it contains many agents...
        '''
        observation = s
        
        observation_ = s_
        action = a[0]
        reward = r[0]
        transition = np.hstack((observation, [action, reward], observation_))

        self.agents[0]["memory"][index, :] = transition
        '''
        
        self.memory_counter += 1

    def choose_action(self, observations):
        actions = []
        # iterating all agents
        agent_index = 0
        for i in range(len(observations)): 
            observation = observations[i]
            # to have batch dimension when feed into tf placeholder
            observation = observation[np.newaxis, :]

            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                #print({self.s: observation})
                actions_value = self.sess.run(self.agents[agent_index]["q_eval"], feed_dict={self.agents[agent_index]["s"]: observation})
                #print("actions_value:")
                #print(actions_value)
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
            actions.append(action)
            agent_index += 1
        
        return actions

    def learn(self):
        

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            

        # iterate every single one of the agent, get its q_target, train its network, and evaluate its cost.
        overall_cost = 0
        for i in range(self.nb_agents):
        
            # check to replace target parameters
            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.agents[i]["replace_target_op"])
                print('\ntarget_params_replaced\n')
        
            batch_memory = self.agents[i]["memory"][sample_index, :]
            #batch_memory = batch_memories[i]
            
            
            q_next, q_eval = self.sess.run(
                [self.agents[i]["q_next"],self.agents[i]["q_eval"]],
                feed_dict={
                    self.agents[i]["s_"]: batch_memory[:, -self.n_features:],  # fixed params
                    self.agents[i]["s"]: batch_memory[:, :self.n_features],  # newest params
                })

            # change q_target w.r.t q_eval's action
            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

            """
            For example in this batch I have 2 samples and 3 actions:
            q_eval =
            [[1, 2, 3],
             [4, 5, 6]]

            q_target = q_eval =
            [[1, 2, 3],
             [4, 5, 6]]

            Then change q_target with the real q_target value w.r.t the q_eval's action.
            For example in:
                sample 0, I took action 0, and the max q_target value is -1;
                sample 1, I took action 2, and the max q_target value is -2:
            q_target =
            [[-1, 2, 3],
             [4, 5, -2]]

            So the (q_target - q_eval) becomes:
            [[(-1)-(1), 0, 0],
             [0, 0, (-2)-(6)]]

            We then backpropagate this error w.r.t the corresponding action to network,
            leave other action as error=0 cause we didn't choose it.
            """

            # train eval network
            _, self.agents[i]["cost"] = self.sess.run([self.agents[i]["_train_op"], self.agents[i]["loss"]],
                                         feed_dict={self.agents[i]["s"]: batch_memory[:, :self.n_features],
                                                    self.agents[i]["q_target"]: q_target})
                                                    
            overall_cost += self.agents[i]["cost"]
            
        # append overall cost for all neural network of all agents    
        self.cost_his.append(overall_cost/self.nb_agents)
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def plot_error(self, errors):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(errors)), errors)
        plt.ylabel('Error')
        plt.xlabel('training steps')
        plt.show()

    def plot_positions(self, xs):
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(len(xs)):
            xis = xs[i]
            for j in range(len(xs[0])):
                xi = xis[j]
                plt.plot(np.arange(len(xi)), xi, label="x"+str(i)+str(j))
           
        #plt.plot(np.arange(len(xs)), xs, label="x_avg")
        plt.legend()
        plt.ylabel('Positions (x)')
        plt.xlabel('training steps')
        plt.show()
