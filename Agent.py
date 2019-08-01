import numpy as np
import random
from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import threading
from time import ctime
import pandas as pd
import tensorflow as tf
global graph
import time
graph = tf.get_default_graph()




class ADQN:
    def __init__(self,n_nodes,node,n_port): #n_nodes = 总路由器数 node = 路由器编号 n_port = 这个路由器有多少个port
        self.n_nodes = n_nodes
        self.node = node
        self.n_port = n_port
        with graph.as_default():
            self.model = Sequential()
            self.model.add(Dense(16, activation='relu',input_shape=(self.n_nodes+1,)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(8,activation='relu'))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(n_port,activation='linear'))
            self.model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
            a = random.randint(1,self.n_nodes)
            state_input = to_categorical(a,num_classes=self.n_nodes+1)
            #state_input[0] = 1
            state_input_array = np.array(state_input)
            state_input_array = state_input_array.reshape(1,self.n_nodes+1)
            self.model.predict(state_input_array)
            self.model._make_train_function()
            weight_name = str(self.node) + '_weights.h5'
            self.model.save_weights(weight_name)
        with graph.as_default():
            self.target_model = Sequential()
            self.target_model.add(Dense(16, activation='relu',input_shape=(self.n_nodes+1,)))
            self.target_model.add(Dropout(0.2))
            self.target_model.add(Dense(8,activation='relu'))
            self.target_model.add(Dropout(0.1))
            self.target_model.add(Dense(n_port, activation='linear'))
            self.target_model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.01), metrics=['accuracy'])
            a = random.randint(1,self.n_nodes)
            state_input = to_categorical(a,num_classes=self.n_nodes+1)
            #state_input[0] = 1
            state_input_array = np.array(state_input)
            state_input_array = state_input_array.reshape(1,self.n_nodes+1)
            self.target_model.predict(state_input_array)
    def estimate(self,dest,receive_port,epsilon):   #返回两个值 第一个值是最小Q值的动作（port）， 第二个是epsilon贪婪算法的动作（port）
        state_input = to_categorical(dest,num_classes=self.n_nodes+1)
        #state_input[0] = 1
        state_input_array = np.array(state_input)
        state_input_array = state_input_array.reshape(1,self.n_nodes+1)
        with graph.as_default():
            Q_estimate_array = self.target_model.predict(state_input_array)
        Q_estimate_list = Q_estimate_array.tolist()
        #print('node',self.node)
        #i = 0
        while True:
            Q_min_port = Q_estimate_list[0].index(min(Q_estimate_list[0])) + 1
            if Q_min_port != receive_port:
                break
            else:
                index = Q_estimate_list[0].index(min(Q_estimate_list[0]))
                Q_estimate_list[0][index] = max(Q_estimate_list[0]) + 1
        #print('node',self.node,'length = Q _es',len(Q_estimate_list[0]))
        p_greedy = 1 - epsilon + epsilon / len(Q_estimate_list[0])
        p_not_greedy = epsilon / len(Q_estimate_list[0])
        p_list = []
        for i in range(1,self.n_port+1):
            if i != Q_min_port:
                p_list.append(p_not_greedy)
            else:
                p_list.append(p_greedy)
        #print(p_list)
        # asd =0
        # for i in range(self.n_port):
        #     asd += p_list[i]
        # print(self.node,Q_estimate_list,Q_min_port,dest,asd)
        p = np.array(p_list)
        port_list = []
        for i in range(self.n_port):
            port_list.append(i+1)
        while True:
            Q_egreddy_port = np.random.choice(port_list, p=p.ravel())
            if Q_egreddy_port != receive_port:
                break
        return Q_min_port,Q_egreddy_port, Q_estimate_list[0]
    def target(self,dest,MinQ_port_eval): #返回target网络的最小值(价值评估)
        weight_name = str(self.node) + '_weights.h5'
        with graph.as_default():
            self.target_model.load_weights(weight_name)
        state_input = to_categorical(dest, num_classes=self.n_nodes+1)
        # state_input[0] = 1
        state_input_array = np.array(state_input)
        state_input_array = state_input_array.reshape(1, self.n_nodes+1)
        with graph.as_default():
            Q_estimate_array = self.target_model.predict(state_input_array)
        Q_estimate_list = Q_estimate_array.tolist()
        Q_min_actual = Q_estimate_list[0][MinQ_port_eval-1]
        return Q_min_actual
    def learn(self,sample_list):
        state_list = []
        port_list = []
        reward_list = []
        MinQ_list = []
        Q_eval_list =[]
        Q_actual_list = []
        impotance_sampling_list = []
        for i in range(len(sample_list)):
            state_list.append(sample_list[i][0])
            port_list.append(sample_list[i][1])
            reward_list.append(sample_list[i][2])
            Q_eval_list.append(sample_list[i][3])
            MinQ_list.append(sample_list[i][4])
            impotance_sampling_list.append((sample_list[i][5]))
        for i in range(len(sample_list)):
            Q_actual_list.append(reward_list[i]+0.9*MinQ_list[i])
        state_input = to_categorical(state_list,num_classes=self.n_nodes+1)
        # for i in range(len(state_input)):
        #     state_input[i][0] = 1
        state_input_array = np.array(state_input)
        state_input_array = state_input_array.reshape(len(state_list),self.n_nodes+1)
        label_list = Q_eval_list
        for i in range(len(sample_list)):
            change_port = port_list[i]
            change_port_index = change_port -1
            label_list[i][change_port_index] = Q_actual_list[i]
        label_list_array = np.array(label_list)
        label_list_array = label_list_array.reshape(len(state_list),len(Q_eval_list[0]))
        sample_weights = np.array(impotance_sampling_list)
        with graph.as_default():
            #print(self.node,'learning start')
            #reduce_lr = ReduceLROnPlateau(monitor='loss',factor=0.1, patience=2, mode='auto')
            self.model.fit(np.array(state_input_array[0]).reshape(1,self.n_nodes+1), np.array(label_list_array[0]).reshape(1,len(Q_eval_list[0])), batch_size=1, epochs=5, verbose=0,sample_weight = np.atleast_1d(sample_weights[0]))
            #print(self.node,'learning end')
        for i in range(1,len(state_input)):
            label = self.model.predict(np.array(state_input_array[i]).reshape(1,self.n_nodes+1))
            # if self.node == 1:
            #     print(state_input_array[i].reshape(1,self.n_nodes+1))
            #     print('ex label',label)
            change_port = port_list[i]
            change_port_index = change_port -1
            label[0][change_port_index] = Q_actual_list[i]
            label = np.array(label)
            label = label.reshape(1,len(Q_eval_list[0]))
            # if self.node == 1:
            #     print('af label',label)
            #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, mode='auto')
            self.model.fit(np.array(state_input_array[i]).reshape(1,self.n_nodes+1), label, batch_size=1, epochs=5, verbose=0)#sample_weight=np.atleast_1d(sample_weights[i]))
        # if self.node == 1:
        #     print('after learning')
        #     test_list = []
        #     for i in range(1,self.n_nodes+1):
        #         if i != self.node:
        #             test_list.append(i)
        #     state_input = to_categorical(test_list, num_classes=self.n_nodes+1)
        #     # for i in range(len(test_list)):
        #     #     state_input[i][0] = 1
        #     state_input_array = np.array(state_input)
        #     state_input_array = state_input_array.reshape(len(state_input), self.n_nodes+1)
        #     Q_estimate_array = self.model.predict(state_input_array)
        #     print(state_input_array)
        #     print(Q_estimate_array)


    def update_network(self,beta):
        weight_name = str(self.node) + '_weights.h5'
        self.model.save_weights(weight_name)
        if beta > 0.1:
            beta -= 0.3
        return beta
    def update_epsilon(self,epsilon):
        if epsilon > 0.1:
            epsilon -= 0.2
        return epsilon
    def show_routing_table(self):
        weight_name = str(self.node) + '_weights.h5'
        with graph.as_default():
            self.target_model.load_weights(weight_name)
        test_list = []
        for i in range(1,self.n_nodes+1):
            if i != self.node:
                test_list.append(i)
        for i in range(len(test_list)):
            state_input = to_categorical(test_list[i], num_classes=self.n_nodes + 1)
            state_input[0] = 1
            state_input_array = np.array(state_input)
            state_input_array = state_input_array.reshape(1, self.n_nodes + 1)
            with graph.as_default():
                Q_estimate_array = self.target_model.predict(state_input_array)
            print(state_input_array)
            print(Q_estimate_array)

# 
#a = 3
# state_input = to_categorical(a,num_classes=7)
# state_input[0] = 1
# state_input_array = np.array(state_input)
# state_input_array = state_input_array.reshape(1,7)
# pre = model.predict(state_input_array)
# #print(pre)
#

# with graph.as_default():
#     model = Sequential()
#     model.add(Dense(16, activation='relu', bias_initializer='zeros', kernel_initializer='uniform', input_shape=(6,)))
#     model.add(Dropout(0.2))
#     model.add(Dense(8, kernel_initializer='uniform', bias_initializer='zeros', activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(2, kernel_initializer='uniform', bias_initializer='zeros', activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
#     weight_name = '1_weights.h5'
#     model.load_weights(weight_name)
#     for layer in model.layers:
#         weights = layer.get_weights()
#         print(weights)
    #
#
# with graph.as_default():
#     model1 = Sequential()
#     model1.add(Dense(16, activation='relu', bias_initializer='zeros', kernel_initializer='uniform', input_shape=(7,)))
#     model1.add(Dropout(0.2))
#     model1.add(Dense(8, kernel_initializer='uniform', bias_initializer='zeros', activation='relu'))
#     model1.add(Dropout(0.1))
#     model1.add(Dense(3, kernel_initializer='uniform', bias_initializer='zeros', activation='relu'))
#     model1.compile(loss='mean_squared_error', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])


#
# def test(model):
#     a = 3
#     state_input = to_categorical(a,num_classes=7)
#     state_input[0] = 1
#     state_input_array = np.array(state_input)
#     state_input_array = state_input_array.reshape(1,7)
#     with graph.as_default():
#         pre = model.predict(state_input_array)
#     print(pre)
#     print('finish')
#
# def test1(model1):
#     a = 3
#     state_input = to_categorical(a,num_classes=7)
#     state_input[0] = 1
#     state_input_array = np.array(state_input)
#     state_input_array = state_input_array.reshape(1,7)
#     with graph.as_default():
#         pre = model1.predict(state_input_array)
#     print(pre)
#     print('finish_1')
#
# th1 = threading.Thread(target=test,args=(model,))
# th2 = threading.Thread(target = test1,args=(model1,))
# th1.start()
# th2.start()
# th1.join()
# th2.join()

# a = ADQN(6,1,3)
# b = ADQN(6,2,3)
# sample_list = [(3,2,0.5,[0.1,0.2,0.05],0.2,0.1),(4,1,1.2,[0.7,0.2,0.3],0.3,0.2),(5,3,0.6,[0.2,0.4,1],0.4,0.6)]
# a.model._make_train_function()
# b.model._make_train_function()
#
# def test1(a):
#     with graph.as_default():
#         pre = a.estimate(2)
#         print('a before',pre[2])
#         a.learn(sample_list)
#         pre = a.estimate(2)
#         print('a after',pre[2])
#
# def test2(b):
#     with graph.as_default():
#         pre = b.estimate(2)
#         print('b before',pre[2])
#         b.learn(sample_list)
#         pre = b.estimate(2)
#         print('b after',pre[2])
#
# th1 = threading.Thread(target=test1,args=(a,))
# th2 = threading.Thread(target = test2,args=(b,))
#
# th1.start()
# th2.start()

# a = ADQN(5,1,2)
# a.show_routing_table()
























